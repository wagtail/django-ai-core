import pytest

from django_ai_core.contrib.index.base import VectorIndex, IndexRegistry


class TestIndexOne(VectorIndex):
    pass


class TestIndexTwo(VectorIndex):
    pass


def test_index_registry_initialization():
    """Test that a new registry starts empty."""
    registry = IndexRegistry()
    assert len(registry._indexes) == 0


def test_index_registry_register():
    """Test registering indexes with the registry."""
    registry = IndexRegistry()

    decorated = registry.register()(TestIndexOne)
    assert decorated is TestIndexOne

    assert "TestIndexOne" in registry._indexes
    assert registry._indexes["TestIndexOne"] is TestIndexOne


def test_index_registry_register_multiple():
    """Test registering multiple indexes with the registry"""
    registry = IndexRegistry()

    registry.register()(TestIndexOne)
    registry.register()(TestIndexTwo)

    assert "TestIndexOne" in registry._indexes
    assert "TestIndexTwo" in registry._indexes
    assert registry._indexes["TestIndexOne"] is TestIndexOne
    assert registry._indexes["TestIndexTwo"] is TestIndexTwo


def test_index_registry_get():
    """Test getting indexes from the registry."""
    registry = IndexRegistry()

    registry.register()(TestIndexOne)
    registry.register()(TestIndexTwo)

    index_class = registry.get("TestIndexOne")
    assert index_class is TestIndexOne


def test_index_registry_get_nonexistent_index():
    """Test getting nonexistent index from registry"""
    registry = IndexRegistry()

    with pytest.raises(KeyError):
        registry.get("NonExistentIndex")


def test_index_registry_list():
    """Test listing all registered indexes."""
    registry = IndexRegistry()
    assert registry.list() == {}

    registry.register()(TestIndexOne)
    registry.register()(TestIndexTwo)

    indexes = registry.list()
    assert len(indexes) == 2
    assert "TestIndexOne" in indexes
    assert "TestIndexTwo" in indexes


def test_index_registry_list_not_mutable():
    """Test registry listing is not mutable"""
    registry = IndexRegistry()
    registry.list()["TestIndexOne"] = TestIndexOne
    assert "TextIndexOne" not in registry.list()
