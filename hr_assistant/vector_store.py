"""Step 4: Store and search embeddings in Qdrant Cloud or local FAISS."""

import os

from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def _qdrant_client() -> QdrantClient:
    """Create a configured Qdrant Cloud client."""
    if not config.QDRANT_URL:
        raise ValueError("QDRANT_URL is not set. Add it to your .env file.")
    if not config.QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is not set. Add it to your .env file.")

    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)


def build_vector_store(chunks):
    """
    Build a Qdrant vector store from document chunks.
    
    Args:
        chunks: List of document chunks with text content.
    
    Returns:
        Qdrant: Vector store object with indexed embeddings.
    """
    logger.info("Building vector store...")
    embeddings = get_embeddings_model()
    client = _qdrant_client()
    embedding_size = len(embeddings.embed_query(chunks[0].page_content))
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=models.Distance.COSINE,
        ),
    )
    vector_store = Qdrant(
        client=client,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embeddings=embeddings,
    )
    vector_store.add_documents(chunks)
    logger.info("Vector store built successfully.")
    return vector_store


def build_local_vector_store(chunks):
    """Build and persist a local FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    logger.info("Building local FAISS vector store at %s", config.VECTOR_STORE_PATH)
    vector_store = FAISS.from_documents(chunks, get_embeddings_model())
    os.makedirs(os.path.dirname(config.VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(config.VECTOR_STORE_PATH)
    logger.info("Local FAISS vector store built successfully.")
    return vector_store


def load_vector_store():
    """
    Connect to an existing Qdrant Cloud collection.

    Returns:
        Qdrant: Connected vector store object.
    """
    logger.info("Loading existing Qdrant Cloud collection: %s", config.QDRANT_COLLECTION_NAME)
    embeddings = get_embeddings_model()
    client = _qdrant_client()
    return Qdrant(
        client=client,
        #url=config.QDRANT_URL,
        #api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
        embeddings=embeddings,
    )


def load_local_vector_store():
    """Load the persisted local FAISS vector store."""
    from langchain_community.vectorstores import FAISS

    logger.info("Loading local FAISS vector store from %s", config.VECTOR_STORE_PATH)
    return FAISS.load_local(
        config.VECTOR_STORE_PATH,
        get_embeddings_model(),
        allow_dangerous_deserialization=True,
    )


def get_or_create_vector_store(chunks):
    """
    Load existing vector store or create a new one from chunks.
    
    Args:
        chunks: List of document chunks (used only if creating new store).
    
    Returns:
        Qdrant or FAISS: Vector store object.
    """
    if not config.QDRANT_URL and not config.QDRANT_API_KEY:
        if os.path.exists(config.VECTOR_STORE_PATH):
            return load_local_vector_store()
        return build_local_vector_store(chunks)

    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise ValueError(
            "Set both QDRANT_URL and QDRANT_API_KEY for Qdrant, "
            "or unset both to use the local FAISS fallback."
        )

    client = _qdrant_client()
    if client.collection_exists(config.QDRANT_COLLECTION_NAME):
        logger.info(
            "Loading existing Qdrant Cloud collection: %s",
            config.QDRANT_COLLECTION_NAME,
        )
        return load_vector_store()

    logger.info(
        "Creating Qdrant Cloud collection: %s",
        config.QDRANT_COLLECTION_NAME,
    )
    return build_vector_store(chunks)


def get_retriever(vector_store, k=None):
    """
    Create a retriever from the vector store for semantic search.
    
    Args:
        vector_store: Qdrant vector store object.
        k: Number of top results to return. Defaults to config.TOP_K_RESULTS.
    
    Returns:
        Retriever: LangChain retriever for document search.
    """
    if k is None:
        k = config.TOP_K_RESULTS
    logger.info(f"Creating retriever with top {k} results.")
    return vector_store.as_retriever(search_kwargs={"k": k})

