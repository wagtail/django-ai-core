"""Test doubles for generative provider resolution.

Importable by normal Python path so they can be referenced from AI_CORE
settings in tests.
"""

from django_ai_core.generative.providers import EmbeddingProvider, GenerativeProvider


class FakeGenerativeProvider(GenerativeProvider):
    """Minimal generative provider for tests."""

    def __init__(self, *, model: str = "fake-model", **extra):
        self.model = model
        self.extra = extra

    def completion(self, prompt, **kwargs):
        return f"completion:{self.model}:{prompt}"

    async def acompletion(self, prompt, **kwargs):
        return self.completion(prompt, **kwargs)


class FakeGenerativeProviderSubclass(FakeGenerativeProvider):
    """Subclass used to test the `expect=` type narrowing."""


class FakeStreamingProvider(FakeGenerativeProvider):
    """Streaming provider for lifecycle tests.

    Yields ``chunks``; if ``fail`` is set, raises it after the chunks; on a
    clean finish it fills ``usage_capture`` from ``usage`` (an (in, out) tuple),
    mimicking any-llm's terminal-chunk usage."""

    def __init__(self, *, chunks=("Hello ", "world"), usage=None, fail=None, **extra):
        super().__init__(**extra)
        self._chunks = chunks
        self._usage = usage
        self._fail = fail

    def _finish(self, usage_capture):
        if self._fail is not None:
            raise self._fail
        if usage_capture is not None and self._usage is not None:
            usage_capture.input_tokens, usage_capture.output_tokens = self._usage

    def stream(self, prompt, *, usage_capture=None, **kwargs):
        for chunk in self._chunks:
            yield chunk
        self._finish(usage_capture)

    async def astream(self, prompt, *, usage_capture=None, **kwargs):
        for chunk in self._chunks:
            yield chunk
        self._finish(usage_capture)


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
