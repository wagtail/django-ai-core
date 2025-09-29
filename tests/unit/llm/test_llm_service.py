import pytest
from unittest import mock
from django_ai_core.llm import LLMService

from any_llm import AnyLLM


class MockAnyLLM(mock.Mock):
    def __init__(self, **kwargs):
        super().__init__(spec=AnyLLM, **kwargs)
        self.PROVIDER_NAME = "mock-provider"


@pytest.fixture
def mock_any_llm():
    return MockAnyLLM()


def test_llm_service_completion_wraps_anyllm(mock_any_llm):
    messages = [
        {
            "role": "user",
            "content": "What is the airspeed velocity of an unladen swallow?",
        }
    ]
    service = LLMService(client=mock_any_llm, model="mock-model")
    service.completion(messages)
    mock_any_llm.completion.assert_called_once_with(
        model="mock-model", messages=messages
    )


def test_llm_service_responses_wraps_anyllm(mock_any_llm):
    prompt = "What is the airspeed velocity of an unladen swallow?"
    service = LLMService(client=mock_any_llm, model="mock-model")
    service.responses(prompt)
    mock_any_llm.responses.assert_called_once_with(
        model="mock-model", input_data=prompt
    )


def test_llm_service_embedding_wraps_anyllm(mock_any_llm):
    prompt = "What is the airspeed velocity of an unladen swallow?"
    service = LLMService(client=mock_any_llm, model="mock-model")
    service.embedding(prompt)
    mock_any_llm._embedding.assert_called_once_with(model="mock-model", inputs=prompt)
