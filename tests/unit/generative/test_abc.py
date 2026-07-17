import asyncio

import pytest

from django_ai_core.generative.providers import GenerativeProvider


class _StubGenerative(GenerativeProvider):
    def completion(self, prompt, **kwargs):
        return "ok"

    async def acompletion(self, prompt, **kwargs):
        return "ok"


def test_generative_provider_cannot_instantiate_directly():
    with pytest.raises(TypeError):
        GenerativeProvider()


def test_stub_generative_provider_works():
    assert _StubGenerative().completion("hi") == "ok"


def test_stream_default_raises_not_implemented():
    stub = _StubGenerative()
    with pytest.raises(NotImplementedError):
        list(stub.stream("hi"))


def test_astream_default_raises_not_implemented():
    stub = _StubGenerative()
    agen = stub.astream("hi")

    async def consume():
        async for _ in agen:
            pass

    with pytest.raises(NotImplementedError):
        asyncio.run(consume())
