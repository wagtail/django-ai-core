import pytest

from django_ai_core.contrib.agents.base import Agent, AgentRegistry


class TestAgentOne(Agent):
    slug = "test-one"
    description = "Test agent one"
    parameters = []


class TestAgentTwo(Agent):
    slug = "test-two"
    description = "Test agent two"
    parameters = []


class AgentWithoutSlug(Agent):
    description = "Agent without slug"
    parameters = []


def test_agent_registry_initialization():
    """Test that a new registry is empty."""
    registry = AgentRegistry()
    assert len(registry._agents) == 0


def test_agent_registry_register_with_slug():
    """Test registering agents with the registry with an explicit slug."""
    registry = AgentRegistry()

    decorated = registry.register("explicit-slug")(TestAgentOne)
    assert decorated is TestAgentOne
    assert "explicit-slug" in registry._agents
    assert registry._agents["explicit-slug"] is TestAgentOne


def test_agent_registry_register_with_agent_slug():
    """Test registering agents with the registry with a slug on the agent class."""
    registry = AgentRegistry()

    registry.register()(TestAgentOne)
    assert "test-one" in registry._agents
    assert registry._agents["test-one"] is TestAgentOne


def test_agent_registry_register_without_slug():
    """Test registering agents with the registry without a slug."""
    registry = AgentRegistry()

    registry.register()(AgentWithoutSlug)
    assert "agentwithoutslug" in registry._agents
    assert registry._agents["agentwithoutslug"] is AgentWithoutSlug


def test_agent_registry_get():
    """Test retrieving agents from the registry."""
    registry = AgentRegistry()

    # Register agents
    registry.register("agent-one")(TestAgentOne)
    registry.register()(TestAgentTwo)

    # Get registered agent
    agent_one = registry.get("agent-one")
    assert agent_one is TestAgentOne


def test_agent_registry_get_nonexistent_agent():
    """Test getting nonexistent agent from registry"""
    registry = AgentRegistry()

    with pytest.raises(KeyError):
        registry.get("non-existent-agent")


def test_agent_registry_list():
    """Test listing all registered agents."""
    registry = AgentRegistry()
    assert registry.list() == {}

    registry.register("one")(TestAgentOne)
    registry.register("two")(TestAgentTwo)

    agents = registry.list()
    assert len(agents) == 2
    assert "one" in agents
    assert "two" in agents


def test_agent_registry_list_not_mutable():
    """Test registry listing is not mutable"""
    registry = AgentRegistry()
    registry.list()["test-agent"] = TestAgentOne
    assert "TextAgentOne" not in registry.list()
