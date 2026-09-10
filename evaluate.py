"""
Run the correctness evaluations and upload results to Langsmith.

Run with: python evaluate.py
"""

from hr_assistant.evaluations import run_evaluation

def main():
    """
    Running the the hr policy assistant evaluations...
    """
    print(" Running the the hr policy assistant evaluations...")

    results = run_evaluation()

    print("Done open your langsmith and see the experiment")
    print(results)

if __name__ == "__main__":
    main()