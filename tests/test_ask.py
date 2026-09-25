"""
Smoke test: confirm ask_assistant(agent, question) works end to end before CI
spends time/money on the full evaluation suite.

Builds the real agent (real Groq / Jina / Qdrant / Portkey calls) and asks it one
question. This is deliberately an integration check, not a mocked unit test.
"""

from hr_assistant.agent import ask_assistant
from hr_assistant.guardrails import GUARD_FAILURE_MESSAGE
from hr_assistant.pipeline import setup_rag_pipeline


def test_ask_returns_a_real_answer():
    agent, _retriever, _vector_store = setup_rag_pipeline()

    answer = ask_assistant(agent, "How many paid sick leave days do I get per year?")

    assert isinstance(answer, str) and answer.strip()
    assert answer != GUARD_FAILURE_MESSAGE
    assert "10" in answer
