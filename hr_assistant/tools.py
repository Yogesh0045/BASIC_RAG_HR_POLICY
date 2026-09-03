""" Step-5: Wrap the retriever as a tool the agent can call. """

from langchain.tools import tool
from hr_assistant.logger import get_logger
from hr_assistant.guardrails import input_safety

logger = get_logger(__name__)


def create_search_tool(retriever):
    """
    Create a tool that wraps the retriever for agent use.
    
    Args:
        retriever: LangChain retriever object from Qdrant Cloud.
    
    Returns:
        Tool: A LangChain Tool object that can be used by agents.
    """
    
    @tool
    def search_hr_policy(question: str) -> str:
        """Search the HR policy document for specific information. Use this tool to look up facts about employee benefits, leave policies, probation, notice periods, reimbursement, code of conduct, holidays, and exit procedures."""
        question = input_safety(question)
        logger.info(f"Searching HR policy for: {question}")
        matching_chunks = retriever.invoke(question)
        logger.info(f"Found %d matching chunk(s)", len(matching_chunks))
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)
    
    return search_hr_policy


def create_tools(retriever):
    """
    Create a list of tools for the agent.
    
    Args:
        retriever: LangChain retriever object.
    
    Returns:
        list: List of Tool objects.
    """
    return [create_search_tool(retriever)]