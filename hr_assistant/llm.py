"""Step-6: Connect to the LLM model ("The brain of the assistant") """

#from langchain_groq import ChatGroq

from hr_assistant.gateway import get_gateway_llm
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    """
    Create and return a ChatOpenAI LLM instance through the Portkey Gateway.
    
    Returns:
        ChatOpenAI: LLM model configured with Portkey Gateway.
    """
    config.check_api_keys()
    logger.info(f"Initializing LLM model: via Portkey Gateway")
    llm = get_gateway_llm()
    return llm
