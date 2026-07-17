"""Provider abstract classes for the generative module.

Two narrow interfaces, split because the underlying jobs are split:
generative completions vs. embeddings. A single concrete provider class
may implement both if a real backend does both jobs.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass
class UsageCapture:
    """Best-effort token-usage sink passed into a provider stream.

    A provider populates whatever usage it can read locally, synchronously
    (no ``await``), at every exit path. Fields stay ``None`` when unavailable —
    e.g. a stream cancelled before its terminal usage frame arrived. The
    consuming service reads this in its own ``finally`` for lifecycle logging.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None


class GenerativeProvider(ABC):
    """Generate text completions from a prompt."""

    #: Human-readable model identifier for audit/lifecycle logging. Providers
    #: set it; ``None`` when the provider doesn't track one.
    model: str | None = None

    @abstractmethod
    def completion(self, prompt: Any, **kwargs: Any) -> Any:
        """Synchronous completion. Return type is provider-defined for now;
        a normalised CompletionResponse lands in PR 2."""

    @abstractmethod
    async def acompletion(self, prompt: Any, **kwargs: Any) -> Any:
        """Async completion."""

    def stream(
        self,
        prompt: Any,
        *,
        usage_capture: UsageCapture | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream text deltas. Optional, but override it whenever the backend
        supports streaming; the default raises NotImplementedError for backends
        that cannot stream.

        ``usage_capture``, when given, is filled best-effort with token usage."""
        raise NotImplementedError(f"{type(self).__name__} does not implement streaming")

    async def astream(
        self,
        prompt: Any,
        *,
        usage_capture: UsageCapture | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async stream of text deltas. Optional, but override it whenever the
        backend supports streaming; the default raises NotImplementedError for
        backends that cannot stream.

        ``usage_capture``, when given, is filled best-effort with token usage."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement async streaming"
        )
        yield  # noqa: unreachable


class EmbeddingProvider(ABC):
    """Embed inputs into vector representations."""

    @abstractmethod
    def embedding(self, input: Any, **kwargs: Any) -> Any:
        """Synchronous embedding."""

    @abstractmethod
    async def aembedding(self, input: Any, **kwargs: Any) -> Any:
        """Async embedding."""
