import subprocess
import sys
from unittest import mock

import pytest
from any_llm import AnyLLM

from django_ai_core.llm import LLMService


class MockAnyLLM(mock.Mock):
    def __init__(self, **kwargs):
        super().__init__(spec=AnyLLM, **kwargs)
        self.PROVIDER_NAME = "mock-provider"


@pytest.fixture
def mock_any_llm():
    return MockAnyLLM()


def test_llm_service_create_passes_kwargs_to_anyllm(mock_any_llm):
    with mock.patch.object(AnyLLM, "create", return_value=mock.Mock()) as mock_create:
        LLMService.create(provider="foo", model="foo", foo="bar")
        mock_create.assert_called_once_with(provider="foo", foo="bar")


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


def test_importing_llm_does_not_import_any_llm():
    # any-llm imports every provider SDK it knows about, so it should only be
    # loaded when an LLMService is actually created.
    code = (
        "import sys\n"
        "import django_ai_core.llm\n"
        "assert 'any_llm' not in sys.modules, 'any_llm was imported'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
