import pytest
from chatbot import ChatBot


@pytest.fixture(scope="module")
def bot():
    b = ChatBot()
    if b.training_data and b.vectorstore is None:
        b._rebuild_vectorstore()
    return b


def test_get_response_returns_non_empty_string(bot):
    answer = bot.get_response("What is machine learning?")
    assert isinstance(answer, str)
    assert answer.strip()
    assert "training data" not in answer.lower() or "knowledge" in answer.lower()


def test_greeting_from_knowledge_base(bot):
    answer = bot.get_response("Hello")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
