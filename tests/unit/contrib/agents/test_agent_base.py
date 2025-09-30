from typing import Annotated

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
    assert agent.parameters
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
    with pytest.raises(ValueError, match="invalid slug"):
        type(
            "InvalidAgent",
            (Agent,),
            {
                "slug": "invalid slug",
                "description": "Agent with invalid slug",
                "parameters": [],
            },
        )


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


def test_agent_parameter_as_dict():
    """Test AgentParameter.as_dict() method."""
    param = AgentParameter(
        name="test_param",
        type=str,
        description="A test parameter",
    )

    result = param.as_dict()

    assert result == {
        "name": "test_param",
        "type": "str",
        "description": "A test parameter",
    }


def test_derive_parameters_from_signature():
    """Test _derive_parameters_from_signature method."""

    class TestAgentWithAnnotations(Agent):
        slug = "test-annotated"
        description = "Test agent with type annotations"

        def execute(
            self,
            *,
            name: Annotated[str, "The name parameter"],
            count: Annotated[int, "The count parameter"],
        ):
            return f"Hello {name}, count is {count}"

    agent = TestAgentWithAnnotations()

    assert agent.parameters
    assert len(agent.parameters) == 2

    name_param = agent.parameters[0]
    assert name_param.name == "name"
    assert name_param.type is str
    assert name_param.description == "The name parameter"

    count_param = agent.parameters[1]
    assert count_param.name == "count"
    assert count_param.type is int
    assert count_param.description == "The count parameter"


def test_derive_parameters_without_descriptions():
    """Test parameter derivation without Annotated descriptions."""

    class TestAgentAnnotatedWithoutDescriptions(Agent):
        slug = "test-no-desc"
        description = "Test agent without descriptions"

        def execute(self, *, value: str, flag: bool):
            return "result"

    agent = TestAgentAnnotatedWithoutDescriptions()

    assert agent.parameters
    assert len(agent.parameters) == 2

    value_param = agent.parameters[0]
    assert value_param.name == "value"
    assert value_param.type is str
    assert value_param.description == ""

    flag_param = agent.parameters[1]
    assert flag_param.name == "flag"
    assert flag_param.type is bool
    assert flag_param.description == ""


def test_explicit_parameters_override_derived():
    """Test that explicit parameters take precedence over derived ones."""

    class AgentWithExplicitParams(Agent):
        slug = "test-explicit"
        description = "Test agent with explicit parameters"
        parameters = [
            AgentParameter(
                name="custom",
                type=str,
                description="Custom parameter",
            ),
        ]

        def execute(self, *, name: Annotated[str, "Should be ignored"]):
            return "result"

    agent = AgentWithExplicitParams()

    assert agent.parameters
    assert len(agent.parameters) == 1
    assert agent.parameters[0].name == "custom"
    assert agent.parameters[0].description == "Custom parameter"


def test_agent_without_parameter_schema():
    class AgentWithNoParams(Agent):
        slug = "test-explicit"
        description = "Test agent with explicit parameters"

        def execute(self, *, name):
            return "result"

    agent = AgentWithNoParams()

    assert not agent.parameters
