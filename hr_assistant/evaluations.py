"""
Step 9: Use the OpenEvals library to evaluate the LLM's performance.
Evaluate the answer quality against a fixed set of questions and answers. 

Unlike tracing (which just records what happened), evaluattion runs the against
a known set of questions and answers pairs and scores each answer using a second LLM as a judge.
Results are uploaded to LangSmith as a dataset + experiment, so quality can be compared across
runs (after a prompt change, a new model, a new guardrail, etc). 
This is a more rigorous way to measure the quality of the LLM's output.

The judge model is routed through the Portkey Gateway, 
using the same slug as the main LLM but a different underlying model (JUDGE_MODEL_NAME) so it isn't 
grading its own output verbatim, without needing a second slug set up. 
This ensures that the evaluation is done in a secure and consistent manner.
"""

from langsmith import Client
from hr_assistant.logger import get_logger

from hr_assistant.agent import create_agent_executor
from hr_assistant.guardrails import input_safety, output_safety
from hr_assistant.llm import get_the_judge_llm
from hr_assistant.vector_store import get_retriever, load_vector_store
from langchain_core.messages import HumanMessage

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT

logger = get_logger(__name__)

DATASET_NAME = "hr_assistant_evaluation"

TEST_CASES = [
    {"question": "How many days of paid annual leave do I get per year?",
    "answer": "You are entitled to 20 days of paid annual leave per year."},
    {"question": "How many days of unused annual leave can I carry over to the next year?",
    "answer": "You can carry over a maximum of 5 days of unused annual leave"},
    {"question": "How many paid sick leave days do I get per year?",
        "answer": "You are entitled to 10 days of paid sick leave per year."},
    {"question": "How many days per week can I work from home?",
         "answer": "Up to 2 days, per week, you can work from home."},
     {"question": "Within how many days must reimbursement claims be submitted?",
      "answer": "Reimbursement claims must be submitted within 30 days of the expense being incurred."},
      {"question" : "How many public holidays are there in a year?",
             "answer": "There are 12 public holidays in a year."},
             {"question": "Within how many days is full and final settlement processed?",
                "answer": "Full and final settlement is processed within 45 days of the last working day."}
]

def _ensure_dataset(client: Client):
    """
    Create the LangSmith dataset if it doesn't exist yet, and load the dataset.
    """
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info(
        "Creating dataset '%s' with %d example(s)",
        DATASET_NAME,
        len(TEST_CASES),
    )
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]}, "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ]
    )

    return dataset

# Write the answer
def run_evaluation():
    """
    Upload the dataset if needed and run the correctness evaluation.
    """
    client = Client()
    dataset = _ensure_dataset(client)

    # Use the existing Qdrant collection; evaluations must not create or load
    # a local vector store.
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    agent = create_agent_executor(retriever)
    judge = get_the_judge_llm()

    # Giving marks
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=judge,
        feedback_key="correctness",
    )
    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        judge=judge,
        feedback_key="groundedness",
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Check that the answer is supported by the retrieved context."""
        return groundedness_judge(
            inputs=kwargs.get("inputs", {}),
            outputs=outputs,
            context=outputs.get("contexts", []),
            reference_outputs=kwargs.get("reference_outputs"),
        )

    def target(inputs):
        question = inputs["question"]
        response = agent.invoke({
            "messages": [HumanMessage(content=input_safety(question))],
        })
        messages = response.get("messages", [])
        answer = output_safety(messages[-1].content if messages else None)
        contexts = [
            message.content
            for message in messages
            if getattr(message, "type", None) == "tool"
        ]
        return {"answer": answer, "contexts": contexts}

    results = client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix="hr-assistant",
    )
    logger.info("Evaluation started for dataset '%s'", DATASET_NAME)
    return results


