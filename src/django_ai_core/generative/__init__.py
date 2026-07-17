"""Generative AI module.

Public API:
- ``GenerativeService`` and ``GenerativeService.for_role(name)`` — the service
  entry point for role-resolved completions.
- ``GenerativeProvider`` / ``EmbeddingProvider`` — abstract classes to subclass
  for a custom provider.
- ``resolve_generative_provider`` / ``resolve_embedding_provider`` — used
  when you need the provider instance directly (e.g. to call vendor-specific
  domain methods on a subclass).
"""

from ..exceptions import (
    AICoreProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
)
from .providers import EmbeddingProvider, GenerativeProvider
from .resolve import resolve_embedding_provider, resolve_generative_provider
from .service import GenerativeService

__all__ = [
    "AICoreProviderError",
    "EmbeddingProvider",
    "GenerativeProvider",
    "GenerativeService",
    "ProviderConfigurationError",
    "ProviderRateLimitError",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderUnexpectedError",
    "resolve_embedding_provider",
    "resolve_generative_provider",
]
