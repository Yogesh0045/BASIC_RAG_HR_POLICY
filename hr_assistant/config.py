""" All settings for the app live here, in one place."""

import os
from dotenv import load_dotenv

load_dotenv()

## ENV VAR/SECRETS

## LLM API KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

## EMBEDDINGS API KEY
JINA_API_KEY = os.getenv("JINA_API_KEY")

## LANGSMITH API KEY
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

## Define the data path and vectore store path
LOAD_FILE_PATH = os.path.join("data", "hr_policy.txt")

## Vector store


## In memory

## Persistant in memory- vectors

## Cloud memory


VECTOR_STORE_PATH = os.path.join("data", "faiss_index")


## Models (LLM and Embedding model)
LLM_MODEL_NAME= "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME= "jina-embeddings-v3"

## CHUNK / TEXT SPLITTING CONFIG

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RETRIEVAL RESULTS
TOP_K_RESULTS = 3

# SYSTEM INSTRUCTIONS

system_prompt= (
   "You are a friendly HR assistant"
    "Always use the search_hr_policy tool to look up"
    "facts before answering. If the answer isn't in the search results,"
    "say you don't know the answer instead of guessing."
)

# Function to check API KEYS

def check_api_keys() -> None:
    """Stop early with a clear message if a required API key is missing"""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set. Add it to your .env file.")