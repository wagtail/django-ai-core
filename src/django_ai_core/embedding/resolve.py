"""Resolve embedding role names to instantiated provider objects."""

from typing import TypeVar

from django_ai_core.resolve import resolve_provider
from django_ai_core.settings import get_embedding_models

from .providers import EmbeddingProvider

E = TypeVar("E", bound=EmbeddingProvider)


def resolve_embedding_provider(
    name: str,
    *,
    expect: type[E] | None = None,
) -> EmbeddingProvider:
    """Resolve a role from ``AI_CORE['EMBEDDING_MODELS']`` to an instance."""
    return resolve_provider(
        name,
        base=EmbeddingProvider,
        models=get_embedding_models(),
        models_key="EMBEDDING_MODELS",
        expect=expect,
    )
