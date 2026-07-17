"""Re-export test doubles for convenient relative import from tests."""

from ._fakes import (
    FakeGenerativeProvider,
    FakeGenerativeProviderSubclass,
    NotAProvider,
)

__all__ = [
    "FakeGenerativeProvider",
    "FakeGenerativeProviderSubclass",
    "NotAProvider",
]
