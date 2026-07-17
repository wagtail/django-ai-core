"""Re-export test doubles for convenient relative import from tests."""

from ._fakes import (
    FakeEmbeddingProvider,
    FakeGenerativeProvider,
    FakeGenerativeProviderSubclass,
    NotAProvider,
)

__all__ = [
    "FakeEmbeddingProvider",
    "FakeGenerativeProvider",
    "FakeGenerativeProviderSubclass",
    "NotAProvider",
]
