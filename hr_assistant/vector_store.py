""" Step-4: Store chunk embeddings in FAISS so we can search them later."""

import os
from langchain_community.vectorstores import FAISS

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model


def build_vector_store(chunks):
    """
    Build a FAISS vector store from document chunks.
    
    Args:
        chunks: List of document chunks with text content.
    
    Returns:
        FAISS: Vector store object with indexed embeddings.
    """
    embeddings = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_vector_store(vector_store, path=None):
    """
    Save the FAISS vector store to disk.
    
    Args:
        vector_store: FAISS vector store object.
        path: Optional path to save. Defaults to config.VECTOR_STORE_PATH.
    """
    if path is None:
        path = config.VECTOR_STORE_PATH
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vector_store.save_local(path)
    print(f"Vector store saved to {path}")


def load_vector_store(path=None):
    """
    Load a FAISS vector store from disk.
    
    Args:
        path: Optional path to load from. Defaults to config.VECTOR_STORE_PATH.
    
    Returns:
        FAISS: Loaded vector store object.
    """
    if path is None:
        path = config.VECTOR_STORE_PATH
    
    embeddings = get_embeddings_model()
    vector_store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


def get_or_create_vector_store(chunks):
    """
    Load existing vector store or create a new one from chunks.
    
    Args:
        chunks: List of document chunks (used only if creating new store).
    
    Returns:
        FAISS: Vector store object.
    """
    path = config.VECTOR_STORE_PATH
    
    if os.path.exists(path):
        print(f"Loading existing vector store from {path}")
        return load_vector_store(path)
    else:
        print("Creating new vector store...")
        vector_store = build_vector_store(chunks)
        save_vector_store(vector_store, path)
        return vector_store


def get_retriever(vector_store, k=None):
    """
    Create a retriever from the vector store for semantic search.
    
    Args:
        vector_store: FAISS vector store object.
        k: Number of top results to return. Defaults to config.TOP_K_RESULTS.
    
    Returns:
        Retriever: LangChain retriever for document search.
    """
    if k is None:
        k = config.TOP_K_RESULTS
    return vector_store.as_retriever(search_kwargs={"k": k})

