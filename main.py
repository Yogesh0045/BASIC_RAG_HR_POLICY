""" 
Command line demo of the HR Policy Assistant.
Run with: python main.py

"""

from hr_assistant import logger
from hr_assistant.pipeline import setup_rag_pipeline
from hr_assistant.agent import ask_assistant
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def main():
    """
    Main entry point for the CLI HR Policy Assistant.
    """
    try:
        # Set up the RAG pipeline
        agent_executor, retriever, vector_store = setup_rag_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("Welcome to the HR Policy Assistant!")
        logger.info("Type 'exit' or 'quit' to end the conversation.")
        logger.info("=" * 60 + "\n")
        
        # Interactive loop for asking questions
        while True:
            question = input("\nYou: ").strip()
            
            if not question:
                logger.info("Please enter a question.")
                continue
            
            if question.lower() in ["exit", "quit"]:
                logger.info("\nThank you for using the HR Policy Assistant. Goodbye!")
                break
            
            # Get the answer from the assistant
            answer = ask_assistant(agent_executor, question)
    
    except KeyboardInterrupt:
        logger.info("\n\nAssistant stopped by user.")
    except Exception as e:
        logger.error(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
