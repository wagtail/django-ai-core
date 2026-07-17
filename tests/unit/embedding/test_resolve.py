import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_core.embedding import resolve_embedding_provider

from ._fakes import FakeEmbeddingProvider

_EMB_FAKE = "embedding._fakes.FakeEmbeddingProvider"


@override_settings(
    AI_CORE={
        "EMBEDDING_MODELS": {
            "default": {"provider": _EMB_FAKE, "params": {"model": "m"}}
        }
    }
)
def test_resolves_configured_role():
    provider = resolve_embedding_provider("default")
    assert isinstance(provider, FakeEmbeddingProvider)
    assert provider.model == "m"


@override_settings(AI_CORE={"EMBEDDING_MODELS": {}})
def test_unknown_role_raises():
    with pytest.raises(ImproperlyConfigured):
        resolve_embedding_provider("missing")
