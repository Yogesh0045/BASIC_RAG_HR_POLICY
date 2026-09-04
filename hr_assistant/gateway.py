
"""
Step 6b: Route the LLM through the Portkey Gateway.

Instead of calling the Groq API directly, the main LLM call goes
through the Portkey Gateway. Portkey stores the real Groq credentials behind
"slug" (set up once in the Portkey dashboard) and allows you to call the Groq API without exposing your credentials.
our code never sees the raw Groq API key, only the Portkey slug. This is a more secure way to call the Groq API.
If the primary slug/model fails, Portkey automatically falls back to a secondary slug/model, if configured. 
This is useful for high availability and redundancy.
"""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_gateway_llm() -> ChatOpenAI:
    """
    Get a ChatOpenAI instance that routes through the Portkey Gateway.
    This is the main model used for the HR assistant.
    """
    logger.info("Creating ChatOpenAI instance with Portkey Gateway routing.")
    headers = createHeaders(
        api_key=config.PORTKEY_API_KEY,
        provider="@hrpolicy",
    )
    return ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        api_key="portkey",  # dummy value, the real key is in the headers
        base_url=PORTKEY_GATEWAY_URL,
        default_headers=headers, # Real key is in the headers, not in the api_key param
        max_tokens=1024,
        request_timeout=30,
    )

