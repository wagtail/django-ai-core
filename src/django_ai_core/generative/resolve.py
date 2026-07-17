"""Resolve role names to instantiated provider objects.

Each resolver call returns a *fresh* instance — no caching at this layer.
Providers are responsible for their own lazy SDK client construction if
they need cheap repeated instantiation.
"""

from typing import TypeVar

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

from .providers import EmbeddingProvider, GenerativeProvider
from .settings import get_embedding_models, get_generative_models

G = TypeVar("G", bound=GenerativeProvider)
E = TypeVar("E", bound=EmbeddingProvider)


def _resolve(
    name: str,
    *,
    models: dict,
    base: type,
    expect: type | None,
    models_key: str,
):
    label = f"AI_CORE['{models_key}']"
    if name not in models:
        raise ImproperlyConfigured(
            f"Role '{name}' not configured in {label}. "
            f"Available roles: {sorted(models)!r}."
        )
    spec = models[name]
    if not isinstance(spec, dict) or "provider" not in spec:
        raise ImproperlyConfigured(
            f"{label}['{name}'] missing 'provider' key. "
            "Expected {'provider': 'dotted.path', 'params': {...}}."
        )
    provider_path = spec["provider"]
    params = spec.get("params", {}) or {}

    try:
        cls = import_string(provider_path)
    except ImportError as exc:
        raise ImproperlyConfigured(
            f"Cannot import provider '{provider_path}' for role '{name}': {exc}"
        ) from exc

    if not (isinstance(cls, type) and issubclass(cls, base)):
        raise ImproperlyConfigured(
            f"Role '{name}' resolves to {cls!r}, expected subclass of {base.__name__}."
        )

    try:
        instance = cls(**params)
    except TypeError as exc:
        raise ImproperlyConfigured(
            f"Cannot instantiate '{provider_path}' for role '{name}' with "
            f"params={params!r}: {exc}"
        ) from exc

    if expect is not None and not isinstance(instance, expect):
        raise ImproperlyConfigured(
            f"Role '{name}' resolves to {type(instance).__name__}, "
            f"expected {expect.__name__}."
        )

    return instance


def resolve_generative_provider(
    name: str,
    *,
    expect: type[G] | None = None,
) -> GenerativeProvider:
    """Resolve a role from ``AI_CORE['GENERATIVE_MODELS']`` to an instance."""
    return _resolve(
        name,
        models=get_generative_models(),
        base=GenerativeProvider,
        expect=expect,
        models_key="GENERATIVE_MODELS",
    )


def resolve_embedding_provider(
    name: str,
    *,
    expect: type[E] | None = None,
) -> EmbeddingProvider:
    """Resolve a role from ``AI_CORE['EMBEDDING_MODELS']`` to an instance."""
    return _resolve(
        name,
        models=get_embedding_models(),
        base=EmbeddingProvider,
        expect=expect,
        models_key="EMBEDDING_MODELS",
    )
