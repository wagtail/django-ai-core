"""Shared any-llm adapter glue: error translation + usage extraction.

Domain-neutral — used by every concrete any-llm-backed provider (generative
and embedding). No completion- or embedding-specific logic lives here. We
enable any-llm's unified exceptions so it normalises every vendor's failures
into its ``AnyLLMError`` hierarchy, which we map to django-ai-core's own
``AICoreProviderError`` types. Consumers never see raw vendor errors or
``AnyLLMError``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

# any-llm only converts vendor SDK errors into its unified ``AnyLLMError``
# hierarchy when this flag is set (otherwise it re-raises the raw vendor error
# with a deprecation warning). We depend on the unified types, so enable it by
# default — ``setdefault`` leaves an explicit consumer choice untouched. Must be
# set before any call; any-llm reads the env var at raise time.
os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

from any_llm.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    GatewayTimeoutError,
    InsufficientFundsError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
    UnsupportedProviderError,
    UpstreamProviderError,
)

from ..exceptions import (
    AICoreProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
)
from ..usage import UsageCapture

# Maps any-llm's unified exceptions to our semantic ``AICoreProviderError`` types.
# any-llm's ``ProviderError`` is a catch-all (transport failures, 5xx,
# unclassified) we can't split without sniffing SDK class names, so it maps whole
# to ``ProviderUnexpectedError`` — as does anything unmatched, via the fallback.
_EXCEPTION_MAP = (
    (RateLimitError, ProviderRateLimitError),
    (GatewayTimeoutError, ProviderTimeoutError),
    (UpstreamProviderError, ProviderUnavailableError),
    (ProviderError, ProviderUnexpectedError),
    (
        (
            AuthenticationError,
            MissingApiKeyError,
            UnsupportedProviderError,
            UnsupportedParameterError,
            InvalidRequestError,
            ModelNotFoundError,
            ContextLengthExceededError,
            InsufficientFundsError,
        ),
        ProviderConfigurationError,
    ),
)


def translate_error(exc: Exception) -> AICoreProviderError:
    """Map a raw any-llm error to a semantic provider error.

    Already-semantic errors pass through. Unified any-llm exceptions map to
    their semantic equivalent; anything else is a real-but-unclassified failure.
    """
    if isinstance(exc, AICoreProviderError):
        return exc
    for exc_types, target in _EXCEPTION_MAP:
        if isinstance(exc, exc_types):
            return target(str(exc))
    return ProviderUnexpectedError(str(exc))


@contextmanager
def translate_errors():
    """Re-raise any raw provider failure as a semantic ``AICoreProviderError``."""
    try:
        yield
    except AICoreProviderError:
        raise
    except Exception as exc:
        raise translate_error(exc) from exc


def fill_usage(capture: UsageCapture, chunk: object) -> None:
    """Copy any usage on this chunk into ``capture`` (best-effort, no raise).

    any-llm sets ``.usage`` (``prompt_tokens`` / ``completion_tokens``) only on
    the terminal chunk of a clean finish. A cancelled stream never reaches it, so
    ``capture`` stays ``None``."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is not None:
        capture.input_tokens = prompt_tokens
    if completion_tokens is not None:
        capture.output_tokens = completion_tokens
