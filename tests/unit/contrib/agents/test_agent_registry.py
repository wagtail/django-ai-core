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


def test_agent_registry_initialization():
    """Test that a new registry is empty."""
    registry = AgentRegistry()
    assert len(registry._agents) == 0


def test_agent_registry_register_uses_agent_slug():
    """Test registering agents with the registry uses the agent's slug attribute."""
    registry = AgentRegistry()

    decorated = registry.register()(TestAgentOne)
    assert decorated is TestAgentOne
    assert "test-one" in registry._agents
    assert registry._agents["test-one"] is TestAgentOne


def test_agent_registry_register_multiple_agents():
    """Test registering multiple agents with the registry."""
    registry = AgentRegistry()

    registry.register()(TestAgentOne)
    registry.register()(TestAgentTwo)

    assert "test-one" in registry._agents
    assert registry._agents["test-one"] is TestAgentOne
    assert "test-two" in registry._agents
    assert registry._agents["test-two"] is TestAgentTwo


def test_agent_registry_get():
    """Test retrieving agents from the registry."""
    registry = AgentRegistry()

    registry.register()(TestAgentOne)
    registry.register()(TestAgentTwo)

    agent_one = registry.get("test-one")
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

    registry.register()(TestAgentOne)
    registry.register()(TestAgentTwo)

    agents = registry.list()
    assert len(agents) == 2
    assert "test-one" in agents
    assert "test-two" in agents


def test_agent_registry_list_not_mutable():
    """Test registry listing is not mutable"""
    registry = AgentRegistry()
    registry.list()["test-agent"] = TestAgentOne
    assert "TextAgentOne" not in registry.list()
