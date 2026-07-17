"""Back-compat re-export. Canonical home: django_ai_core.prompt."""

import warnings

from django_ai_core.prompt import Prompt, TokenDict

warnings.warn(
    "Importing from django_ai_core.llm.prompt is deprecated and will be removed "
    "in a future release. Import from django_ai_core.prompt instead: "
    "`from django_ai_core.prompt import Prompt, TokenDict`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Prompt", "TokenDict"]
