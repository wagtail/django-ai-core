import warnings

from .base import LLMService

__all__ = ["LLMService", "Prompt", "TokenDict"]

# Prompt / TokenDict moved to django_ai_core.prompt. Keep them reachable here for
# back-compat, but resolve lazily via __getattr__ so importing LLMService does
# not drag in — or warn about — the prompt re-export.
_MOVED = {"Prompt", "TokenDict"}


def __getattr__(name):
    if name in _MOVED:
        warnings.warn(
            f"Importing {name} from django_ai_core.llm is deprecated and will be "
            "removed in a future release. Import from django_ai_core.prompt "
            f"instead: `from django_ai_core.prompt import {name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        from django_ai_core import prompt

        return getattr(prompt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
