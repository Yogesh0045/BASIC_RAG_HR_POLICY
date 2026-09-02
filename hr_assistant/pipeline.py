""" Wires all the all components together into ready-to-use agent.
This is the single entry point that main.py (CLI) and app.py (streamlite)
both call. Each step is handled by its own small module.

"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config
from hr_assistant.vector_store import (
    build_vector_store,
    save_vector_store,
    get_or_create_vector_store,
    get_retriever,
)
from hr_assistant.agent import create_agent_executor
from hr_assistant.logger import get_logger  

logger = get_logger(__name__)

def load_documents():
    """
    Load documents from the configured file path.
    
    Returns:
        list: List of loaded documents.
    """
    loader = TextLoader(config.LOAD_FILE_PATH, encoding="utf-8")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document(s)")
    return documents


def split_documents(documents):
    """
    Split documents into chunks using recursive character splitter.
    
    Args:
        documents: List of documents to split.
    
    Returns:
        list: List of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks


def setup_rag_pipeline():
    """
    Set up the complete RAG pipeline.
    
    Orchestrates all steps:
    1. Load documents
    2. Split into chunks
    3. Build/load vector store
    4. Create retriever
    5. Build agent executor
    
    Returns:
        tuple: (agent_executor, retriever, vector_store)
    """
    logger.info("\n" + "=" * 60)
    logger.info("Setting up HR RAG Pipeline...")
    logger.info("=" * 60 + "\n")
    
    # Step 1: Load documents
    logger.info("Step 1: Loading documents...")
    documents = load_documents()
    
    # Step 2: Split into chunks
    logger.info("Step 2: Splitting documents into chunks...")
    chunks = split_documents(documents)
    
    # Step 3: Build/load vector store
    logger.info("Step 3: Setting up vector store...")
    vector_store = get_or_create_vector_store(chunks)
    logger.info(f"Vector store ready with {vector_store.index.ntotal} embeddings")
    
    # Step 4: Create retriever
    logger.info("\nStep 4: Creating retriever...")
    retriever = get_retriever(vector_store)
    logger.info(f"Retriever configured for top-{config.TOP_K_RESULTS} results")
    
    # Step 5: Build agent
    logger.info("\nStep 5: Building agent executor...")
    agent_executor = create_agent_executor(retriever)
    logger.info("Agent ready!")
    
    logger.info("\n" + "=" * 60)
    logger.info("HR RAG Pipeline Setup Complete!")
    logger.info("=" * 60 + "\n")
    
    return agent_executor, retriever, vector_store
