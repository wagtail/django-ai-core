"""Read the AI_CORE Django setting with explicit, distinguishable errors."""

from typing import Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


def get_ai_core_setting(key: str) -> Any:
    """Return ``settings.AI_CORE[key]``.

    Raises ``ImproperlyConfigured`` if ``AI_CORE`` is absent or the key is
    missing.
    """
    ai_core = getattr(settings, "AI_CORE", None)
    if ai_core is None:
        raise ImproperlyConfigured(
            "AI_CORE setting is not configured. Add an AI_CORE dict to your "
            "Django settings (see django_ai_core docs)."
        )
    if not isinstance(ai_core, dict):
        raise ImproperlyConfigured(
            f"AI_CORE setting must be a dict, got {type(ai_core).__name__}."
        )
    if key not in ai_core:
        raise ImproperlyConfigured(
            f"AI_CORE['{key}'] is not configured. Add '{key}' to your AI_CORE "
            "settings dict."
        )
    return ai_core[key]


def _get_models_dict(key: str) -> dict:
    value = get_ai_core_setting(key)
    if not isinstance(value, dict):
        raise ImproperlyConfigured(
            f"AI_CORE['{key}'] must be a dict mapping role names to provider "
            f"specs, got {type(value).__name__}."
        )
    return value


def get_generative_models() -> dict:
    """Return the configured generative model role map."""
    return _get_models_dict("GENERATIVE_MODELS")


def get_embedding_models() -> dict:
    """Return the configured embedding model role map."""
    return _get_models_dict("EMBEDDING_MODELS")
