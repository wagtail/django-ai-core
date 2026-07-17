"""Generic role-name → provider-instance resolution.

Domain packages wrap this with their own ABC and settings key. Duck-typed on
``base=`` so the kernel never imports a domain ABC.

Each call returns a *fresh* instance — no caching at this layer. Providers are
responsible for their own lazy SDK client construction if they need cheap
repeated instantiation.
"""

from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string


def resolve_provider(
    name: str,
    *,
    base: type,
    models: dict,
    models_key: str,
    expect: type | None = None,
):
    """Resolve ``name`` from ``models`` (an ``AI_CORE[models_key]`` map) to an
    instance of a ``base`` subclass. ``expect`` optionally narrows the accepted
    concrete type."""
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
