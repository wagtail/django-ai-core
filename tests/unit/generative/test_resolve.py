import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_core.generative.resolve import resolve_generative_provider

from ._fakes import (
    FakeGenerativeProvider,
    FakeGenerativeProviderSubclass,
)

_GEN_FAKE = "generative._fakes.FakeGenerativeProvider"
_GEN_FAKE_SUB = "generative._fakes.FakeGenerativeProviderSubclass"
_NOT_PROVIDER = "generative._fakes.NotAProvider"


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {
            "default": {"provider": _GEN_FAKE, "params": {"model": "m1"}}
        }
    }
)
def test_resolve_generative_returns_instance():
    provider = resolve_generative_provider("default")
    assert isinstance(provider, FakeGenerativeProvider)
    assert provider.model == "m1"


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {
            "default": {"provider": _GEN_FAKE, "params": {"model": "m1"}}
        }
    }
)
def test_resolve_returns_fresh_instance_each_call():
    a = resolve_generative_provider("default")
    b = resolve_generative_provider("default")
    assert a is not b


@override_settings(AI_CORE={"GENERATIVE_MODELS": {}})
def test_resolve_unknown_role_raises():
    with pytest.raises(ImproperlyConfigured, match="Role 'default' not configured"):
        resolve_generative_provider("default")


@override_settings(AI_CORE={"GENERATIVE_MODELS": {"default": {"params": {}}}})
def test_resolve_missing_provider_key_raises():
    with pytest.raises(ImproperlyConfigured, match="missing 'provider' key"):
        resolve_generative_provider("default")


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {
            "default": {"provider": "nonexistent.module.Class", "params": {}}
        }
    }
)
def test_resolve_import_error_raises():
    with pytest.raises(ImproperlyConfigured, match="Cannot import provider"):
        resolve_generative_provider("default")


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {"default": {"provider": _NOT_PROVIDER, "params": {}}}
    }
)
def test_resolve_non_provider_class_raises():
    with pytest.raises(
        ImproperlyConfigured, match="expected subclass of GenerativeProvider"
    ):
        resolve_generative_provider("default")


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {
            "default": {"provider": _GEN_FAKE, "params": {"model": "m1"}}
        }
    }
)
def test_resolve_expect_matches():
    p = resolve_generative_provider("default", expect=FakeGenerativeProvider)
    assert isinstance(p, FakeGenerativeProvider)


@override_settings(
    AI_CORE={
        "GENERATIVE_MODELS": {
            "default": {"provider": _GEN_FAKE, "params": {"model": "m1"}}
        }
    }
)
def test_resolve_expect_mismatch_raises():
    with pytest.raises(ImproperlyConfigured, match="expected"):
        resolve_generative_provider("default", expect=FakeGenerativeProviderSubclass)
