import asyncio
from types import SimpleNamespace

import pytest
from any_llm.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    GatewayTimeoutError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

from django_ai_core.generative import (
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
)
from django_ai_core.generative.providers.anyllm import (
    AnyLLMProvider,
    _delta_text,
    _message_text,
    build_messages,
)
from django_ai_core.providers.anyllm import translate_error as _translate
from django_ai_core.providers.anyllm import translate_errors as _translate_errors
from django_ai_core.usage import UsageCapture

# --- build_messages -------------------------------------------------------


def test_build_messages_single_user_prompt():
    assert build_messages("Hello", system=None, history=None) == [
        {"role": "user", "content": "Hello"},
    ]


def test_build_messages_prepends_system():
    msgs = build_messages("Hi", system="Be terse.", history=None)
    assert msgs[0] == {"role": "system", "content": "Be terse."}
    assert msgs[-1] == {"role": "user", "content": "Hi"}


def test_build_messages_includes_history_then_prompt():
    history = [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
    ]
    msgs = build_messages("next", system="S", history=history)
    assert msgs == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "next"},
    ]


# --- unified any-llm exception mapping ------------------------------------

# any-llm exception -> expected semantic type.
MAPPING_CASES = [
    (RateLimitError, ProviderRateLimitError),
    (GatewayTimeoutError, ProviderTimeoutError),
    (UpstreamProviderError, ProviderUnavailableError),
    (AuthenticationError, ProviderConfigurationError),
    (InvalidRequestError, ProviderConfigurationError),
    (ModelNotFoundError, ProviderConfigurationError),
    (ContextLengthExceededError, ProviderConfigurationError),
    # catch-all: transport, 5xx, unclassified -> unexpected.
    (ProviderError, ProviderUnexpectedError),
]


@pytest.mark.parametrize(
    ("exc_cls", "expected"),
    MAPPING_CASES,
    ids=[c.__name__ for c, _ in MAPPING_CASES],
)
def test_translate_unified_error(exc_cls, expected):
    assert isinstance(_translate(exc_cls("boom")), expected)


def test_translate_missing_api_key_is_configuration():
    # MissingApiKeyError keeps a positional (provider, env_var) constructor.
    exc = MissingApiKeyError("anthropic", "ANTHROPIC_API_KEY")
    assert isinstance(_translate(exc), ProviderConfigurationError)


def test_translate_unknown_is_unexpected_error():
    assert isinstance(_translate(ValueError("weird")), ProviderUnexpectedError)


def test_translate_passes_through_semantic_error():
    original = ProviderRateLimitError("already mapped")
    assert _translate(original) is original


# --- context manager ------------------------------------------------------


def test_translate_errors_wraps_and_chains():
    src = RateLimitError("rate limited")
    with pytest.raises(ProviderRateLimitError) as caught, _translate_errors():
        raise src
    assert caught.value.__cause__ is src


def test_translate_errors_passes_through_semantic():
    with pytest.raises(ProviderResponseError), _translate_errors():
        raise ProviderResponseError("keep me")


# --- response-shape contract ----------------------------------------------
#
# Build the *real* any-llm types (not our fakes) so an upgrade that changed the
# response shape fails here rather than silently returning empty text.


def test_message_text_reads_real_chat_completion():
    resp = ChatCompletion.model_validate(
        {
            "id": "x",
            "model": "m",
            "object": "chat.completion",
            "created": 0,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello!"},
                }
            ],
        }
    )
    assert _message_text(resp) == "Hello!"


def test_delta_text_reads_real_chat_completion_chunk():
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "model": "m",
            "object": "chat.completion.chunk",
            "created": 0,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
        }
    )
    assert _delta_text(chunk) == "Hel"


def test_delta_text_tolerates_contentless_chunk():
    """A chunk with no text delta (e.g. a role-only or finish chunk) yields ''."""
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "x",
            "model": "m",
            "object": "chat.completion.chunk",
            "created": 0,
            "choices": [{"index": 0, "delta": {}}],
        }
    )
    assert _delta_text(chunk) == ""


# --- provider integration -------------------------------------------------
#
# Exercise the four AnyLLMProvider methods against a fake any-llm client,
# confirming the translation wrapper is applied (classification rules covered
# above).


def _completion_response(text):
    """The shape any-llm returns for a non-streaming completion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _delta_chunk(text):
    """The shape any-llm yields for one streaming chunk."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


def _usage_chunk(prompt_tokens, completion_tokens):
    """The terminal chunk any-llm yields on clean finish, carrying usage.

    OpenAI-shaped: ``.usage.prompt_tokens`` / ``.usage.completion_tokens``.
    The delta is contentless (usage rides the finish frame)."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


def _rate_limit_error():
    """A representative raw any-llm error (unified rate-limit)."""
    return RateLimitError("rate limited")


class _FakeClient:
    """Stand-in for an any-llm client. Each test injects the call behaviour it
    needs: `completion` / `acompletion` are plain callables (a function, a
    generator, or an async coroutine, depending on what's under test)."""

    def __init__(self, *, completion=None, acompletion=None):
        if completion is not None:
            self.completion = completion
        if acompletion is not None:
            self.acompletion = acompletion


def _make_provider(client):
    """An AnyLLMProvider wired to a fake client.

    __init__ is bypassed deliberately: the real one calls ``AnyLLM.create()``,
    which needs vendor credentials and network. We only want to test error
    translation, so we inject the fake client and the two attributes the
    methods read (`_provider`, `_model`)."""
    provider = AnyLLMProvider.__new__(AnyLLMProvider)
    provider._provider = "anthropic"
    provider.model = "test-model"
    provider._client = client
    return provider


def test_completion_returns_text():
    """A successful completion returns the extracted message text."""
    provider = _make_provider(
        _FakeClient(completion=lambda **kw: _completion_response("Hello!"))
    )
    assert provider.completion("hi") == "Hello!"


def test_completion_translates_sdk_error():
    """A raw vendor error from a non-streaming call surfaces as the semantic type."""

    def boom(**kwargs):
        raise _rate_limit_error()

    provider = _make_provider(_FakeClient(completion=boom))
    with pytest.raises(ProviderRateLimitError):
        provider.completion("hi")


def test_completion_empty_response_raises_response_error():
    """A successful-but-empty completion is treated as an unusable response."""
    provider = _make_provider(
        _FakeClient(completion=lambda **kw: _completion_response(""))
    )
    with pytest.raises(ProviderResponseError):
        provider.completion("hi")


def test_acompletion_returns_text():
    """The async non-streaming path awaits the client and returns text (-> str)."""

    async def acompletion(**kwargs):
        return _completion_response("Hello async!")

    provider = _make_provider(_FakeClient(acompletion=acompletion))
    assert asyncio.run(provider.acompletion("hi")) == "Hello async!"


def test_acompletion_translates_sdk_error():
    """A raw vendor error from the async non-streaming call is translated."""

    async def acompletion(**kwargs):
        raise _rate_limit_error()

    provider = _make_provider(_FakeClient(acompletion=acompletion))
    with pytest.raises(ProviderRateLimitError):
        asyncio.run(provider.acompletion("hi"))


def test_stream_translates_sdk_error_mid_iteration():
    """An error raised after some chunks have streamed is still translated."""

    def stream(**kwargs):
        yield _delta_chunk("Hi")
        raise _rate_limit_error()

    provider = _make_provider(_FakeClient(completion=stream))
    with pytest.raises(ProviderRateLimitError):
        list(provider.stream("hi"))


# --- best-effort usage capture -------------------------------------------


def test_stream_fills_usage_capture_from_terminal_chunk():
    def stream(**kwargs):
        yield _delta_chunk("Hi")
        yield _usage_chunk(10, 5)

    provider = _make_provider(_FakeClient(completion=stream))
    cap = UsageCapture()
    text = "".join(provider.stream("hi", usage_capture=cap))
    assert text == "Hi"
    assert cap.input_tokens == 10
    assert cap.output_tokens == 5


def test_stream_usage_capture_stays_none_without_terminal_chunk():
    def stream(**kwargs):
        yield _delta_chunk("Hi")

    provider = _make_provider(_FakeClient(completion=stream))
    cap = UsageCapture()
    list(provider.stream("hi", usage_capture=cap))
    assert cap.input_tokens is None
    assert cap.output_tokens is None


def test_stream_without_usage_capture_still_streams():
    def stream(**kwargs):
        yield _delta_chunk("Hi")
        yield _usage_chunk(10, 5)

    provider = _make_provider(_FakeClient(completion=stream))
    assert "".join(provider.stream("hi")) == "Hi"


def test_astream_fills_usage_capture_from_terminal_chunk():
    async def acompletion(**kwargs):
        async def _chunks():
            yield _delta_chunk("Hi")
            yield _usage_chunk(7, 3)

        return _chunks()

    provider = _make_provider(_FakeClient(acompletion=acompletion))
    cap = UsageCapture()

    async def _consume():
        return "".join([c async for c in provider.astream("hi", usage_capture=cap)])

    assert asyncio.run(_consume()) == "Hi"
    assert cap.input_tokens == 7
    assert cap.output_tokens == 3


def test_astream_translates_sdk_error():
    """The async streaming path translates errors too."""

    async def acompletion(**kwargs):
        async def _chunks():
            raise _rate_limit_error()
            yield  # unreachable; makes this an async generator

        return _chunks()

    provider = _make_provider(_FakeClient(acompletion=acompletion))

    async def _consume():
        async for _ in provider.astream("hi"):
            pass

    with pytest.raises(ProviderRateLimitError):
        asyncio.run(_consume())
