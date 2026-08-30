"""step-7: Build the agent ties the LLM and the search tool together."""

from langgraph.prebuilt import create_react_agent

from hr_assistant import config
from hr_assistant.llm import get_llm
from hr_assistant.tools import create_tools


def create_agent_executor(retriever):
    """
    Create a complete agent executor with tools and LLM.
    
    Args:
        retriever: FAISS retriever from vector store.
    
    Returns:
        A compiled graph agent ready to process queries.
    """
    llm = get_llm()
    tools = create_tools(retriever)
    
    # Create a React agent using langgraph
    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=config.system_prompt,
    )
    
    return agent_executor


def ask_assistant(agent_executor, question: str) -> str:
    """
    Ask the HR assistant a question.
    
    Args:
        agent_executor: Compiled graph agent instance.
        question: Question to ask the assistant.
    
    Returns:
        str: The assistant's response.
    """
    from langchain_core.messages import HumanMessage
    
    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    
    response = agent_executor.invoke({
        "messages": [HumanMessage(content=question)],
    })
    
    # Extract the last message from the response
    messages = response.get("messages", [])
    answer = messages[-1].content if messages else "No answer found"
    
    print("ANSWER:", answer)
    print("=" * 60)
    
    return answer
