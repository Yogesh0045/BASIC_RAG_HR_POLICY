""" Step-1: Read the raw document from the data folder. """

from langchain_community.document_loaders import TextLoader
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def load_documents(file_path: str = config.DATA_FILE_PATH):
    """Load a .txt file and return it as a list of LangChain Document """
    logger.info(f"Loading document from {file_path}")
    documents = TextLoader(file_path, encoding="utf-8")
    logger.info(f"Loaded %d documents(s)", len(documents))

    return documents.load()