"""Provider abstract class for the generative module."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any

from django_ai_core.usage import UsageCapture


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
        yield  # pragma: no cover
