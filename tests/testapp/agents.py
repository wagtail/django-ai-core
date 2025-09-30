from django_ai_core.contrib.agents import Agent, registry
from typing import Annotated


@registry.register()
class BasicAgent(Agent):
    slug = "basic"
    description = "Basic agent that just takes a prompt and returns a response."

    def execute(self, *, prompt: Annotated[str, "The prompt to use for the agent"]):
        return prompt


@registry.register()
class StubAgent(Agent):
    slug = "stub"
    description = "Basic agent that just takes a prompt and returns a response."

    def execute(self):
        return ""
