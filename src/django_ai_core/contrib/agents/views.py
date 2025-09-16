from typing import Any, Type
import json

from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from . import registry, Agent


@method_decorator(csrf_exempt, name="dispatch")
class AgentExecutionView(View):
    agent_slug: str = ""

    def post(self, request):
        """
        Execute the agent with provided input data.

        Expected JSON payload:
        {
            "arguments": {
                "foo": "bar"
            }
        }
        """
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in request body"}, status=400)

        if "arguments" not in data:
            return JsonResponse(
                {"error": "Missing required field: arguments"}, status=400
            )

        try:
            agent = registry.get(self.agent_slug)
        except KeyError:
            return JsonResponse(
                {
                    "error": f"Agent not found: {self.agent_slug}",
                    "code": "agent_not_found",
                },
                status=404,
            )

        result = self._execute_agent(agent, data["arguments"])

        return JsonResponse({"status": "completed", "data": result})

    def _execute_agent(self, agent_cls: Type[Agent], arguments: dict[str, Any]) -> Any:
        agent = agent_cls()

        return agent.execute(**arguments)
