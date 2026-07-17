import pytest

from django_ai_core.exceptions import (
    AICoreProviderError,
    ProviderConfigurationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUnexpectedError,
)

REQUEST_LEAVES = [
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnexpectedError,
]


@pytest.mark.parametrize("cls", REQUEST_LEAVES)
def test_request_leaves_are_request_errors(cls):
    assert issubclass(cls, ProviderRequestError)
    assert issubclass(cls, AICoreProviderError)


def test_config_error_is_not_a_request_error():
    assert issubclass(ProviderConfigurationError, AICoreProviderError)
    assert not issubclass(ProviderConfigurationError, ProviderRequestError)


def test_message():
    exc = ProviderUnavailableError("at capacity")
    assert str(exc) == "at capacity"


def test_cause_chaining():
    root = ValueError("boom")

    def raiser():
        try:
            raise root
        except ValueError as e:
            raise ProviderUnexpectedError("net down") from e

    with pytest.raises(ProviderUnexpectedError) as exc_info:
        raiser()
    assert exc_info.value.__cause__ is root


def test_exceptions_are_exported_from_package():
    import django_ai_core.generative as pkg

    names = [
        "AICoreProviderError",
        "ProviderConfigurationError",
        "ProviderRequestError",
        "ProviderTimeoutError",
        "ProviderUnavailableError",
        "ProviderRateLimitError",
        "ProviderResponseError",
        "ProviderUnexpectedError",
    ]
    for name in names:
        assert name in pkg.__all__, f"{name} missing from __all__"
        assert hasattr(pkg, name), f"{name} not importable from package"
