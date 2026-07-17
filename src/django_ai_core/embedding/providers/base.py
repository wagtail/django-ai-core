"""Provider abstract class for the embedding module."""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Embed inputs into vector representations."""

    #: Human-readable model identifier for audit/lifecycle logging. Providers
    #: set it; ``None`` when the provider doesn't track one.
    model: str | None = None

    @abstractmethod
    def embedding(self, input: Any, **kwargs: Any) -> Any:
        """Synchronous embedding."""

    @abstractmethod
    async def aembedding(self, input: Any, **kwargs: Any) -> Any:
        """Async embedding."""
