"""Test doubles for embedding provider resolution."""

from django_ai_core.embedding import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, model: str = "fake-embed", **extra):
        self.model = model
        self.extra = extra

    def embedding(self, input, **kwargs):
        return [[0.0] for _ in input]

    async def aembedding(self, input, **kwargs):
        return self.embedding(input, **kwargs)


class NotAProvider:
    """Used to test the abstract-class-subclass check."""

    def __init__(self, **kwargs):
        pass
