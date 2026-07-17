import asyncio
import logging

import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_core.generative import GenerativeService

from ._fakes import FakeGenerativeProvider, FakeStreamingProvider

_GEN_FAKE = "generative._fakes.FakeGenerativeProvider"

_LIFECYCLE_LOGGER = "django_ai_core.generative.service"


@pytest.fixture
def lifecycle_caplog(caplog):
    """caplog wired to the lifecycle logger, which has ``propagate=False`` in
    test settings so the default root-attached caplog handler never sees it."""
    logger = logging.getLogger(_LIFECYCLE_LOGGER)
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger=_LIFECYCLE_LOGGER)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _lifecycle_record(caplog):
    """The single lifecycle log record emitted by a stream."""
    records = [r for r in caplog.records if r.name == _LIFECYCLE_LOGGER]
    assert len(records) == 1, f"expected 1 lifecycle record, got {len(records)}"
    return records[0]


def test_service_programmatic_constructor():
    provider = FakeGenerativeProvider(model="m1")
    service = GenerativeService(provider=provider)
    assert service.provider is provider
    assert service.role is None
    assert service.completion("hi") == "completion:m1:hi"


def test_service_carries_role_name():
    provider = FakeGenerativeProvider(model="m1")
    service = GenerativeService(provider=provider, role="fast")
    assert service.role == "fast"


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {
            "fast": {"provider": _GEN_FAKE, "params": {"model": "m2"}}
        }
    }
)
def test_for_role_returns_service_backed_by_role():
    service = GenerativeService.for_role("fast")
    assert isinstance(service.provider, FakeGenerativeProvider)
    assert service.role == "fast"
    assert service.provider.model == "m2"


@override_settings(AI_CORE={"GENERATIVE_MODELS": {}})
def test_for_role_unknown_raises():
    with pytest.raises(ImproperlyConfigured, match="Role 'fast' not configured"):
        GenerativeService.for_role("fast")


def test_service_acompletion():
    provider = FakeGenerativeProvider(model="m1")
    service = GenerativeService(provider=provider)
    result = asyncio.run(service.acompletion("hi"))
    assert result == "completion:m1:hi"


def test_service_stream_not_implemented_bubbles_up():
    provider = FakeGenerativeProvider(model="m1")
    service = GenerativeService(provider=provider)
    with pytest.raises(NotImplementedError):
        list(service.stream("hi"))


# --- streaming lifecycle logging ------------------------------------------


def test_stream_forwards_text_unchanged():
    provider = FakeStreamingProvider(chunks=("Hello ", "world"))
    service = GenerativeService(provider=provider)
    assert "".join(service.stream("hi")) == "Hello world"


def test_stream_logs_completed_with_metadata(lifecycle_caplog):
    provider = FakeStreamingProvider(
        model="m1", chunks=("Hello ", "world"), usage=(12, 4)
    )
    service = GenerativeService(provider=provider, role="fast")
    list(service.stream("hi"))
    rec = _lifecycle_record(lifecycle_caplog)
    assert rec.state == "completed"
    assert rec.role == "fast"
    assert rec.model == "m1"
    assert rec.chunks == 2
    assert rec.chars == len("Hello world")
    assert rec.words == 2
    assert rec.input_tokens == 12
    assert rec.output_tokens == 4


def test_stream_logs_cancelled_on_early_close(lifecycle_caplog):
    provider = FakeStreamingProvider(chunks=("a", "b", "c"), usage=(9, 9))
    service = GenerativeService(provider=provider)
    gen = service.stream("hi")
    assert next(gen) == "a"
    gen.close()  # client disconnect → GeneratorExit at the paused yield
    rec = _lifecycle_record(lifecycle_caplog)
    assert rec.state == "cancelled"
    assert rec.chunks == 1
    # terminal usage never reached on cancel
    assert rec.input_tokens is None
    assert rec.output_tokens is None


def test_stream_logs_errored_and_reraises(lifecycle_caplog):
    provider = FakeStreamingProvider(chunks=("a",), fail=RuntimeError("boom"))
    service = GenerativeService(provider=provider)
    with pytest.raises(RuntimeError, match="boom"):
        list(service.stream("hi"))
    rec = _lifecycle_record(lifecycle_caplog)
    assert rec.state == "errored"
    assert rec.chunks == 1


def test_astream_logs_completed_with_metadata(lifecycle_caplog):
    provider = FakeStreamingProvider(model="am1", chunks=("Hi ", "there"), usage=(3, 2))
    service = GenerativeService(provider=provider)

    async def _consume():
        return "".join([c async for c in service.astream("hi")])

    assert asyncio.run(_consume()) == "Hi there"
    rec = _lifecycle_record(lifecycle_caplog)
    assert rec.state == "completed"
    assert rec.model == "am1"
    assert rec.chunks == 2
    assert rec.input_tokens == 3
    assert rec.output_tokens == 2


def test_astream_logs_cancelled_on_aclose(lifecycle_caplog):
    provider = FakeStreamingProvider(chunks=("a", "b", "c"))
    service = GenerativeService(provider=provider)

    async def _run():
        agen = service.astream("hi")
        assert await agen.__anext__() == "a"
        await agen.aclose()  # → GeneratorExit inside the async generator

    asyncio.run(_run())
    rec = _lifecycle_record(lifecycle_caplog)
    assert rec.state == "cancelled"


def test_astream_logs_cancelled_on_cancellederror(lifecycle_caplog):
    provider = FakeStreamingProvider(chunks=("a",), fail=asyncio.CancelledError())
    service = GenerativeService(provider=provider)

    async def _consume():
        async for _ in service.astream("hi"):
            pass

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_consume())
    rec = _lifecycle_record(lifecycle_caplog)
    assert rec.state == "cancelled"
