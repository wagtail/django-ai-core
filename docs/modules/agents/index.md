# Agents

Agents are AI tools which do a specific task in your application. The Django AI Core Agent Framework allows you to register agents which can then either be called directly in your application code, or via a URL on your frontend.

!!! note "Future plans"

    Right now agents aren't much different from views, but the hope is that eventually by registering an agent you will get benefits like:

    - Logging
    - Rate limiting
    - Auth
    - Streaming responses

## Registering an Agent

Register your agents in your application using the Agent registry:

```python
# agents.py
from django_ai_core.contrib.agents import Agent, AgentParameter, registry
from django_ai_core.llm import LLMService


@registry.register()
class SimplePromptAgent(Agent):
    slug = "prompt"
    name = "Basic agent"
    description = "Basic agent that just takes a prompt and returns a response from the LLM."
    parameters = [
        AgentParameter(
            name="prompt",
            type=str,
            description="The prompt to use for the agent",
        ),
    ]

    def execute(self, *, prompt: str):
        service = LLMService.create(provider="openai", model="gpt-4o")
        response = service.completion(prompt)

```

These can be in any file you want, but if you use a file like `agents.py`, you'll need to make sure Django loads it in your `apps.py`:

```python

from django.apps import AppConfig


class MyAppAppConfig(AppConfig):
    label = "myapp"
    name = "mayapp"
    verbose_name = "My example app"

    def ready(self):
        from . import agents
```

## Configuring Agent URLs

Agents can be made accessible at automatically generated URLs by adding `agent_urls()` to your `urlpatterns`:

```python
from django.urls import path, include
from django_ai_core.contrib.agents.urls import agent_urls as ai_core_agent_urls

urlpatterns = [
    ...
    path("ai/", include(ai_core_agent_urls())),
]
```

This will generate URLs under `ai/` for all your registered agents based on their `slug` values. The `SimplePromptAgent` from the above example will be accessible at `ai/prompt/`.

Alternatively, you can register views for individual agents using the `as_view` classmethod on Agent classes:

````python

from django.urls import path, include
from .agents import SimplePromptAgent

urlpatterns = [
    ...
    path("ai/simple/", SimplePromptAgent.as_view()),
]

## Using Agents

Agents can either be invoked directly:

```python
from .agents import SimplePromptAgent

SimplePromptAgent().execute(prompt="Foo")
````

or via their URL:

```
POST https://www.example.com/ai/prompt/

{
    "arguments": {
        "prompt": "Are you suggesting coconuts migrate?"
    }
}
```

## Permissions

You can control who can execute your agents using permission. See the [Permissions](./permissions) documentation for detailed information on how to secure your agents.

Quick example:

```python

from django_ai_core.contrib.agents import Agent
from django_ai_core.contrib.agents.permissions import IsAuthenticated

@registry.register()
class SecureAgent(Agent):
    slug = "secure"
    name = "Secure Agent"
    description = "Agent requiring authentication"
    permission = IsAuthenticated()
    parameters = []

    def execute(self):
        return "Authenticated access only!"
```
