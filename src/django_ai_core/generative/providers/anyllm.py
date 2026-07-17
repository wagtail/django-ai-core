"""Concrete ``GenerativeProvider`` backed by any-llm.

Shipped with django-ai-core as a batteries-included provider. ``any-llm-sdk``
is a core dependency. Error handling is provider-agnostic: we enable any-llm's
unified exceptions (``ANY_LLM_UNIFIED_EXCEPTIONS``) so it normalises every
vendor's failures into its ``AnyLLMError`` hierarchy, which we then map to
django-ai-core's own ``AICoreProviderError`` types. Consumers only ever see our
exceptions — never raw vendor errors, never ``AnyLLMError``. No vendor SDK is
imported here.

Talking to a specific vendor still needs that vendor's client installed, pulled
via an optional extra, e.g. ``django-ai-core[anthropic]`` / ``[openai]``.

Reference by dotted path from ``AI_CORE['GENERATIVE_MODELS']``::

    "provider": "django_ai_core.generative.providers.anyllm.AnyLLMProvider",
    "params": {"provider": "anthropic", "model": "claude-haiku-4-5"},
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from any_llm import AnyLLM

from django_ai_core.providers.anyllm import fill_usage, translate_errors
from django_ai_core.usage import UsageCapture

from ...exceptions import ProviderResponseError
from .base import GenerativeProvider


def build_messages(
    prompt: object,
    *,
    system: str | None,
    history: list[dict] | None,
) -> list[dict]:
    """Assemble an OpenAI-style messages list from (system, history, prompt)."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": str(prompt)})
    return messages


# any-llm normalises every provider's response to the OpenAI-shaped
# ``ChatCompletion`` / ``ChatCompletionChunk``, so the extraction paths below are
# vendor-uniform: ``.choices[0].message.content`` for a completion,
# ``.choices[0].delta.content`` for a stream chunk. test_anyllm asserts this
# against real any-llm types.
def _delta_text(chunk: object) -> str:
    """Extract incremental text from a streaming chunk, tolerating gaps."""
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def _message_text(response: object) -> str:
    return response.choices[0].message.content or ""


def _require_content(text: str) -> str:
    if not text:
        raise ProviderResponseError("provider returned an empty response")
    return text


class AnyLLMProvider(GenerativeProvider):
    """Generative provider over any-llm. Translates failures to the
    ``AICoreProviderError`` hierarchy so consumers never see raw SDK errors."""

    def __init__(self, *, provider: str, model: str, **client_kwargs):
        self._provider = provider
        self.model = model
        self._client = AnyLLM.create(provider=provider, **client_kwargs)

    def completion(self, prompt, *, system=None, history=None, **kwargs) -> str:
        messages = build_messages(prompt, system=system, history=history)
        with translate_errors():
            resp = self._client.completion(
                model=self.model, messages=messages, stream=False, **kwargs
            )
        return _require_content(_message_text(resp))

    async def acompletion(self, prompt, *, system=None, history=None, **kwargs) -> str:
        messages = build_messages(prompt, system=system, history=history)
        with translate_errors():
            resp = await self._client.acompletion(
                model=self.model, messages=messages, stream=False, **kwargs
            )
        return _require_content(_message_text(resp))

    def stream(
        self,
        prompt,
        *,
        system=None,
        history=None,
        usage_capture: UsageCapture | None = None,
        **kwargs,
    ) -> Iterator[str]:
        messages = build_messages(prompt, system=system, history=history)
        with translate_errors():
            for chunk in self._client.completion(
                model=self.model, messages=messages, stream=True, **kwargs
            ):
                if usage_capture is not None:
                    fill_usage(usage_capture, chunk)
                text = _delta_text(chunk)
                if text:
                    yield text

    async def astream(
        self,
        prompt,
        *,
        system=None,
        history=None,
        usage_capture: UsageCapture | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        messages = build_messages(prompt, system=system, history=history)
        with translate_errors():
            stream = await self._client.acompletion(
                model=self.model, messages=messages, stream=True, **kwargs
            )
            async for chunk in stream:
                if usage_capture is not None:
                    fill_usage(usage_capture, chunk)
                text = _delta_text(chunk)
                if text:
                    yield text
