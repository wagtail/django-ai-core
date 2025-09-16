from django_ai_core.contrib.agents import Agent, AgentParameter, registry


@registry.register()
class BasicAgent(Agent):
    name = "basic"
    description = "Basic agent that just takes a prompt and returns a response."
    parameters = [
        AgentParameter(
            name="prompt",
            type=str,
            description="The prompt to use for the agent",
        ),
    ]

    def execute(self, *, prompt: str):
        return prompt
