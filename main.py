""" 
Command line demo of the HR Policy Assistant.
Run with: python main.py

"""

from hr_assistant.pipeline import setup_rag_pipeline
from hr_assistant.agent import ask_assistant


def main():
    """
    Main entry point for the CLI HR Policy Assistant.
    """
    try:
        # Set up the RAG pipeline
        agent_executor, retriever, vector_store = setup_rag_pipeline()
        
        print("\n" + "=" * 60)
        print("Welcome to the HR Policy Assistant!")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("=" * 60 + "\n")
        
        # Interactive loop for asking questions
        while True:
            question = input("\nYou: ").strip()
            
            if not question:
                print("Please enter a question.")
                continue
            
            if question.lower() in ["exit", "quit"]:
                print("\nThank you for using the HR Policy Assistant. Goodbye!")
                break
            
            # Get the answer from the assistant
            answer = ask_assistant(agent_executor, question)
    
    except KeyboardInterrupt:
        print("\n\nAssistant stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
