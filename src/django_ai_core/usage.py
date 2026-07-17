"""Provider-agnostic token-usage sink, shared by every provider layer."""

from dataclasses import dataclass


@dataclass
class UsageCapture:
    """Best-effort token-usage sink passed into a provider stream.

    A provider populates whatever usage it can read locally, synchronously
    (no ``await``), at every exit path. Fields stay ``None`` when unavailable —
    e.g. a stream cancelled before its terminal usage frame arrived. The
    consuming service reads this in its own ``finally`` for lifecycle logging.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
