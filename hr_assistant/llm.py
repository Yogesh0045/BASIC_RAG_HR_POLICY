"""Step-6: Connect to the LLM model ("The brain of the assistant") """

from langchain_groq import ChatGroq
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """
    Create and return a ChatGroq LLM instance.
    
    Validates that GROQ_API_KEY is set before initializing.
    
    Returns:
        ChatGroq: LLM model configured with Groq API.
    """
    config.check_api_keys()
    logger.info(f"Initializing LLM model: {config.LLM_MODEL_NAME}")
    llm = ChatGroq(
        model=config.LLM_MODEL_NAME,
        temperature=0,
    )
    
    return llm
