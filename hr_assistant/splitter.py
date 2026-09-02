"""Step 2: Split the document into small, searchable chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from hr_assistant import config

from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def split_into_chunks(documents):

    """ 
    Split documents into small overlapping chunks

    """
    logger.info(f"Splitting documents into chunks of size {config.CHUNK_SIZE} with overlap {config.CHUNK_OVERLAP}")
    text_spliiter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP
    )
    logger.info(f"Splitting %d document(s) into chunks", len(documents))

    return text_spliiter.split_documents(documents)