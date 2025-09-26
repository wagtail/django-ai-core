import pytest

from django_ai_core.contrib.agents.base import Agent, AgentParameter


class TestAgent(Agent):
    slug = "test-agent"
    description = "Test agent for testing"
    parameters = [
        AgentParameter(
            name="name",
            type=str,
            description="Name parameter",
        ),
        AgentParameter(
            name="count",
            type=int,
            description="Count parameter",
        ),
    ]

    def execute(self, name, count=0):
        return f"Hello {name}, count is {count}"


def test_agent_initialization():
    """Test basic agent initialization."""
    agent = TestAgent()

    assert agent.slug == "test-agent"
    assert agent.description == "Test agent for testing"
    assert len(agent.parameters) == 2

    # Check parameter definitions
    name_param = agent.parameters[0]
    count_param = agent.parameters[1]

    assert name_param.name == "name"
    assert name_param.type is str
    assert name_param.description == "Name parameter"

    assert count_param.name == "count"
    assert count_param.type is int
    assert count_param.description == "Count parameter"


def test_agent_execution():
    """Test agent execution with parameters."""
    agent = TestAgent()

    # Test with required params
    result = agent.execute("Alice")
    assert result == "Hello Alice, count is 0"

    # Test with all params
    result = agent.execute("Bob", count=5)
    assert result == "Hello Bob, count is 5"


def test_agent_invalid_slug():
    """Test that agents with invalid slugs raise exceptions."""
    # We're testing the validation in Agent.__init_subclass__, which runs when the class is defined
    # So we need to dynamically create a class to trigger the validation
    with pytest.raises(ValueError) as excinfo:
        type(
            "InvalidAgent",
            (Agent,),
            {
                "slug": "invalid slug",
                "description": "Agent with invalid slug",
                "parameters": [],
            },
        )

    assert "invalid slug" in str(excinfo.value).lower()


def test_agent_parameter_dataclass():
    """Test the AgentParameter dataclass."""
    param = AgentParameter(
        name="test",
        type=bool,
        description="A test parameter",
    )

    assert param.name == "test"
    assert param.type is bool
    assert param.description == "A test parameter"
