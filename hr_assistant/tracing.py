"""
Step 0b: Langsmith tracing setup.

Langsmith is a tool for tracing and monitoring LangChain applications. This needs no wiring in the codebase, 
just looks for a few environment variables to be set.   
Such as LANGSMITH_TRACING, LANGSMITH_ENDPOINT, LANGSMITH_API_KEY, and LANGSMITH_PROJECT (Loaded from .env by config.py).
If tracing is enabled, it will automatically capture and send traces of every LLM call, tool call, and agent steps to your 
Langsmith project.

This module doesn't turn tracing on -the env vars already do that.
All it does is log, once per run, whether tracing is enabled or not, and what the env vars are set to.
So its obvious from the logs (see logger.py) weather this run was being traced or not, and what the env vars were set to.

"""

import hr_assistant.config as config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def check_langsmith_tracing() -> None:
    """
    Check if Langsmith tracing is enabled and log the status.
    """
    tracing_enabled = config.LANGSMITH_TRACING.lower() == "true"

    if tracing_enabled and not config.LANGSMITH_API_KEY:
        logger.warning("Langsmith tracing is enabled, but LANGSMITH_API_KEY is not set. Tracing will not work.")
    elif tracing_enabled:
        logger.info("Langsmith tracing is ENABLED.")
        logger.info(f"Langsmith Endpoint: {config.LANGSMITH_ENDPOINT}")
        logger.info(f"Langsmith Project: {config.LANGSMITH_PROJECT}")
    else:
        logger.info("Langsmith tracing is DISABLED.")