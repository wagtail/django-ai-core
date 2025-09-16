from dataclasses import dataclass

from django.core.validators import validate_slug
from django.core.exceptions import ValidationError


@dataclass
class AgentParameter:
    name: str
    type: type
    description: str


class Agent:
    """Base class for agents."""

    slug: str
    description: str
    parameters: list[AgentParameter]

    def execute(self, *args, **kwargs) -> str:
        """Execute the agent."""
        return ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if hasattr(cls, "slug"):
            try:
                validate_slug(cls.slug)
            except ValidationError:
                raise ValueError(
                    f"Agent {cls.__name__} has an invalid slug: {cls.slug}. Use a valid “slug” consisting of letters, numbers, underscores or hyphens."
                )


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, type[Agent]] = {}

    def register(self, slug: str | None = None):
        """Decorator to register an agent."""

        def decorator(cls: type[Agent]) -> type[Agent]:
            agent_slug = slug or getattr(cls, "slug", cls.__name__.lower())
            self._agents[agent_slug] = cls
            return cls

        return decorator

    def get(self, slug: str) -> type[Agent]:
        if slug not in self._agents:
            raise KeyError(f"Agent '{slug}' not found")
        return self._agents[slug]

    def list(self) -> dict[str, type[Agent]]:
        return self._agents.copy()


registry = AgentRegistry()
