from types import SimpleNamespace

import pytest
from any_llm.exceptions import ProviderError, RateLimitError

from django_ai_core.exceptions import (
    AICoreProviderError,
    ProviderRateLimitError,
    ProviderUnexpectedError,
)
from django_ai_core.providers.anyllm import (
    fill_usage,
    translate_error,
    translate_errors,
)
from django_ai_core.usage import UsageCapture


def test_translate_maps_rate_limit():
    assert isinstance(translate_error(RateLimitError("x")), ProviderRateLimitError)


def test_translate_catch_all_maps_to_unexpected():
    assert isinstance(translate_error(ProviderError("x")), ProviderUnexpectedError)


def test_translate_passes_through_semantic():
    err = ProviderRateLimitError("keep me")
    assert translate_error(err) is err


def test_translate_errors_ctx_wraps_raw():
    with pytest.raises(AICoreProviderError), translate_errors():
        raise RateLimitError("boom")


def test_fill_usage_reads_terminal_chunk():
    cap = UsageCapture()
    chunk = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=7))
    fill_usage(cap, chunk)
    assert (cap.input_tokens, cap.output_tokens) == (3, 7)


def test_fill_usage_no_usage_is_noop():
    cap = UsageCapture()
    fill_usage(cap, SimpleNamespace())
    assert (cap.input_tokens, cap.output_tokens) == (None, None)
