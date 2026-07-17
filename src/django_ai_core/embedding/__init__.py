"""Embedding module.

Public API:
- ``EmbeddingProvider`` — abstract class to subclass for a custom provider.
- ``resolve_embedding_provider`` — resolve a configured embedding role to an
  instance.
"""

from .providers import EmbeddingProvider
from .resolve import resolve_embedding_provider

__all__ = ["EmbeddingProvider", "resolve_embedding_provider"]
