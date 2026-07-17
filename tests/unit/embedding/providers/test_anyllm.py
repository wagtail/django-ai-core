import asyncio

import pytest

import django_ai_core.embedding.providers.anyllm as mod
from django_ai_core.embedding import EmbeddingProvider
from django_ai_core.embedding.providers.anyllm import AnyLLMEmbeddingProvider
from django_ai_core.exceptions import ProviderRateLimitError


def test_is_embedding_provider():
    assert issubclass(AnyLLMEmbeddingProvider, EmbeddingProvider)


def test_embedding_calls_any_llm_with_provider_and_model(monkeypatch):
    seen = {}

    def fake_embedding(model, inputs, *, provider, **kwargs):
        seen.update(model=model, inputs=inputs, provider=provider, kwargs=kwargs)
        return "RESP"

    monkeypatch.setattr(mod.any_llm, "embedding", fake_embedding)
    p = AnyLLMEmbeddingProvider(provider="openai", model="text-embed", api_key="k")
    out = p.embedding(["a", "b"])
    assert out == "RESP"
    assert seen["model"] == "text-embed"
    assert seen["inputs"] == ["a", "b"]
    assert seen["provider"] == "openai"
    assert seen["kwargs"]["api_key"] == "k"


def test_embedding_translates_errors(monkeypatch):
    from any_llm.exceptions import RateLimitError

    def boom(*a, **k):
        raise RateLimitError("slow down")

    monkeypatch.setattr(mod.any_llm, "embedding", boom)
    p = AnyLLMEmbeddingProvider(provider="openai", model="m")
    with pytest.raises(ProviderRateLimitError):
        p.embedding(["a"])


def test_aembedding_calls_any_llm(monkeypatch):
    async def fake_aembedding(model, inputs, *, provider, **kwargs):
        return "ARESP"

    monkeypatch.setattr(mod.any_llm, "aembedding", fake_aembedding)
    p = AnyLLMEmbeddingProvider(provider="openai", model="m")
    assert asyncio.run(p.aembedding(["a"])) == "ARESP"
