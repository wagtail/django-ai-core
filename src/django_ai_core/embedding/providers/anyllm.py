"""Concrete ``EmbeddingProvider`` backed by any-llm.

Shipped with django-ai-core as a batteries-included provider. Error handling
is provider-agnostic via the shared any-llm glue: consumers only ever see
``AICoreProviderError`` types. Talking to a specific vendor still needs that
vendor's client installed via an optional extra, e.g. ``[openai]``.

Reference by dotted path from ``AI_CORE['EMBEDDING_MODELS']``::

    "provider": "django_ai_core.embedding.providers.anyllm.AnyLLMEmbeddingProvider",
    "params": {"provider": "openai", "model": "text-embedding-3-small"},
"""

from __future__ import annotations

from typing import Any

import any_llm

from django_ai_core.providers.anyllm import translate_errors

from .base import EmbeddingProvider


class AnyLLMEmbeddingProvider(EmbeddingProvider):
    """Embedding provider over any-llm. Translates failures to the
    ``AICoreProviderError`` hierarchy so consumers never see raw SDK errors.

    Returns any-llm's raw ``CreateEmbeddingResponse`` for now; a normalised
    response type can land later without changing this contract's call shape.
    """

    def __init__(self, *, provider: str, model: str, **client_kwargs: Any):
        self._provider = provider
        self.model = model
        self._client_kwargs = client_kwargs

    def embedding(self, input: Any, **kwargs: Any) -> Any:
        with translate_errors():
            return any_llm.embedding(
                self.model,
                input,
                provider=self._provider,
                **self._client_kwargs,
                **kwargs,
            )

    async def aembedding(self, input: Any, **kwargs: Any) -> Any:
        with translate_errors():
            return await any_llm.aembedding(
                self.model,
                input,
                provider=self._provider,
                **self._client_kwargs,
                **kwargs,
            )
