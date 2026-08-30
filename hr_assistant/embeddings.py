""" Step-3: Turn text into numbers (vectors) using jina embedding model. """

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embeddings_model():

    """
    Return a Jina embeddings model. Reads JINA_API_KEY
    from the environment
    
    """
    
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)