"""Service entry point: generic role-resolved completions."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .providers import GenerativeProvider
from .providers.base import UsageCapture
from .resolve import resolve_generative_provider

logger = logging.getLogger(__name__)


class GenerativeService:
    """Sealed-by-convention orchestrator around a ``GenerativeProvider``.

    Use ``GenerativeService.for_role(name)`` for settings-driven config, or
    construct directly with ``GenerativeService(provider=...)`` for tests and
    per-tenant configuration.

    Custom domain methods belong on a provider subclass, not on this class.
    See the generative module docs for the access patterns (service, provider
    subclass, direct provider).
    """

    def __init__(
        self,
        *,
        provider: GenerativeProvider,
        role: str | None = None,
    ):
        self._provider = provider
        self._role = role

    @property
    def provider(self) -> GenerativeProvider:
        return self._provider

    @property
    def role(self) -> str | None:
        return self._role

    @classmethod
    def for_role(cls, name: str) -> "GenerativeService":
        """Resolve a role from settings and return a service backed by it."""
        provider = resolve_generative_provider(name)
        return cls(provider=provider, role=name)

    def completion(self, prompt: Any, **kwargs: Any) -> Any:
        return self._provider.completion(prompt, **kwargs)

    async def acompletion(self, prompt: Any, **kwargs: Any) -> Any:
        return await self._provider.acompletion(prompt, **kwargs)

    def stream(self, prompt: Any, **kwargs: Any) -> Iterator[str]:
        """Stream text deltas, emitting a lifecycle log line on every exit path.

        Wraps the provider stream so that on normal completion, client
        disconnect (``GeneratorExit``), or provider error, a single terminal
        log record is emitted with the run's metadata and best-effort usage."""
        started = time.monotonic()
        usage = UsageCapture()
        chunks = chars = words = 0
        state: str | None = None
        try:
            for text in self._provider.stream(prompt, usage_capture=usage, **kwargs):
                chunks += 1
                chars += len(text)
                words += len(text.split())
                yield text
            state = "completed"
        except GeneratorExit:
            state = "cancelled"
            raise
        except Exception:
            state = "errored"
            raise
        finally:
            self._log_lifecycle(state, started, chunks, chars, words, usage)

    async def astream(self, prompt: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Async counterpart to ``stream``. Cancellation surfaces as either
        ``GeneratorExit`` (``aclose``) or ``asyncio.CancelledError`` (task
        cancel); both map to ``cancelled``."""
        started = time.monotonic()
        usage = UsageCapture()
        chunks = chars = words = 0
        state: str | None = None
        try:
            async for text in self._provider.astream(
                prompt, usage_capture=usage, **kwargs
            ):
                chunks += 1
                chars += len(text)
                words += len(text.split())
                yield text
            state = "completed"
        except (GeneratorExit, asyncio.CancelledError):
            state = "cancelled"
            raise
        except Exception:
            state = "errored"
            raise
        finally:
            self._log_lifecycle(state, started, chunks, chars, words, usage)

    def _log_lifecycle(
        self,
        state: str | None,
        started: float,
        chunks: int,
        chars: int,
        words: int,
        usage: UsageCapture,
    ) -> None:
        logger.info(
            "generative stream %s",
            state,
            extra={
                "state": state,
                "role": self._role,
                "model": self._provider.model,
                "chunks": chunks,
                "chars": chars,
                "words": words,
                "elapsed_s": round(time.monotonic() - started, 3),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )
