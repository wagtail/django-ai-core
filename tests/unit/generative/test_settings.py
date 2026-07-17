import pytest
from django.core.exceptions import ImproperlyConfigured
from django.test import override_settings

from django_ai_core.generative.settings import (
    get_ai_core_setting,
    get_embedding_models,
    get_generative_models,
)


@override_settings(
    AI_CORE={"GENERATIVE_MODELS": {"default": {"provider": "x", "params": {}}}}
)
def test_get_ai_core_setting_returns_value():
    assert get_ai_core_setting("GENERATIVE_MODELS") == {
        "default": {"provider": "x", "params": {}}
    }


@override_settings(AI_CORE={})
def test_get_ai_core_setting_missing_key_raises():
    with pytest.raises(ImproperlyConfigured, match="AI_CORE\\['GENERATIVE_MODELS'\\]"):
        get_ai_core_setting("GENERATIVE_MODELS")


def test_get_ai_core_setting_no_ai_core_attr_raises(settings):
    if hasattr(settings, "AI_CORE"):
        del settings.AI_CORE
    with pytest.raises(ImproperlyConfigured, match="AI_CORE"):
        get_ai_core_setting("GENERATIVE_MODELS")


@override_settings(AI_CORE={"GENERATIVE_MODELS": "not a dict"})
def test_get_generative_models_rejects_non_dict():
    with pytest.raises(ImproperlyConfigured, match="must be a dict"):
        get_generative_models()


@override_settings(
    AI_CORE={"EMBEDDING_MODELS": {"default": {"provider": "x", "params": {}}}}
)
def test_get_embedding_models_returns_dict():
    assert "default" in get_embedding_models()
