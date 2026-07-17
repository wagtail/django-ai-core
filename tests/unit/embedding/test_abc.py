import pytest

from django_ai_core.embedding.providers import EmbeddingProvider


class _StubEmbedding(EmbeddingProvider):
    def embedding(self, input, **kwargs):
        return [[0.0]]

    async def aembedding(self, input, **kwargs):
        return [[0.0]]


def test_embedding_provider_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        EmbeddingProvider()


def test_stub_embedding_provider_works():
    assert _StubEmbedding().embedding(["hi"]) == [[0.0]]
