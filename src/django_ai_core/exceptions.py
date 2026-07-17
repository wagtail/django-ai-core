"""Provider-agnostic exception hierarchy for the whole library.

Shared by every provider layer — generative completions, embeddings, and any
future backend — because they call the same kinds of backends and hit the same
failure modes. Names are semantic, not transport-specific: a concrete provider
maps its own SDK/HTTP failures into this vocabulary from the outside. The
library imports no provider SDK.

Configuration errors surfaced at settings-resolution time keep using Django's
``ImproperlyConfigured`` (raised where a role is resolved to a provider);
``ProviderConfigurationError`` here is for misconfiguration discovered at call
time.
"""

from __future__ import annotations


class AICoreProviderError(Exception):
    """Base for any provider-layer failure (generative or embedding)."""


class ProviderConfigurationError(AICoreProviderError):
    """Runtime provider misconfiguration: bad/missing credentials, unknown
    model, unsupported parameter. Caller error."""


class ProviderRequestError(AICoreProviderError):
    """A real request attempt against the backend failed."""


class ProviderTimeoutError(ProviderRequestError):
    """The request deadline was exceeded."""


class ProviderUnavailableError(ProviderRequestError):
    """Backend reached but refusing: overloaded or at capacity."""


class ProviderRateLimitError(ProviderRequestError):
    """Throttled or quota exhausted by the provider."""


class ProviderResponseError(ProviderRequestError):
    """A response was received but is unusable: empty completion text,
    malformed or wrong-dimension embedding vector, safety-filtered output."""


class ProviderUnexpectedError(ProviderRequestError):
    """A request failed for an unclassified reason: the backend returned an
    error we can't map to a more specific type, or an unexpected exception was
    raised while talking to it. Use when the failure is real but its kind is
    unknown — never over-claim timeout/unavailable/bad-response."""
