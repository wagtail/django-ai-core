import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Callable, TypeVar, get_args, get_origin

from django.core.exceptions import ValidationError
from django.core.validators import validate_slug

from .permissions import BasePermission
from .views import AgentExecutionView

AgentT = TypeVar("AgentT", bound="Agent")


@dataclass
class AgentParameter:
    name: str
    type: type
    description: str
    permission = BasePermission | None

    def as_dict(self):
        return {
            "name": self.name,
            "type": self.type.__name__,
            "description": self.description,
        }


class Agent(ABC):
    """Base class for agents."""

    slug: str
    name: str = ""
    description: str = ""
    parameters: list[AgentParameter] | None = None

    @classmethod
    def as_view(cls):
        return AgentExecutionView.as_view(agent_slug=cls.slug)

    @classmethod
    def _derive_parameters_from_signature(cls) -> list[AgentParameter]:
        """Derive parameters from `execute` type signature"""
        parameters = []
        annotations = inspect.get_annotations(cls.execute)
        for name, annotation in annotations.items():
            if name == "return":
                continue
            description: str = ""
            base_type = annotation
            if get_origin(annotation) is Annotated:
                base_type, *metadata = get_args(annotation)
                if metadata and isinstance(metadata[0], str):
                    description = metadata[0]
            parameters.append(
                AgentParameter(name=name, type=base_type, description=description)
            )
        return parameters

    @abstractmethod
    def execute(self, *args, **kwargs) -> str | dict | list:
        """Execute the agent"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if "parameters" not in cls.__dict__:
            cls.parameters = cls._derive_parameters_from_signature()

        if hasattr(cls, "slug"):
            try:
                validate_slug(cls.slug)
            except ValidationError as e:
                raise ValueError(
                    f"Agent {cls.__name__} has an invalid slug: {cls.slug}. Use a valid “slug” consisting of letters, numbers, underscores or hyphens."
                ) from e


class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, type[Agent]] = {}

    def register(
        self, cls: type[AgentT] | None = None
    ) -> type[AgentT] | Callable[[type[AgentT]], type[AgentT]]:
        def decorator(agent_cls: type[AgentT]) -> type[AgentT]:
            agent_slug = agent_cls.slug
            self._agents[agent_slug] = agent_cls
            return agent_cls

        if cls is None:
            # Called with parentheses: @registry.register()
            return decorator
        else:
            # Called without parentheses: @registry.register
            return decorator(cls)

    def get(self, slug: str) -> type[Agent]:
        if slug not in self._agents:
            raise KeyError(f"Agent '{slug}' not found")
        return self._agents[slug]

    def list(self) -> dict[str, type[Agent]]:
        return self._agents.copy()


registry = AgentRegistry()
