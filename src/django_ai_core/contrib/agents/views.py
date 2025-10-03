import json
from typing import TYPE_CHECKING, Any

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from .permissions import AllowAny

if TYPE_CHECKING:
    from . import Agent


class AgentExecutionException(Exception):
    pass


class AgentNotFound(AgentExecutionException):
    code = "agent_not_found"


@method_decorator(csrf_exempt, name="dispatch")
class AgentExecutionView(View):
    agent_slug: str = ""

    def get(self, request):
        try:
            agent = self._get_agent()
        except AgentNotFound as e:
            return JsonResponse(
                {
                    "error": f"Agent not found: {self.agent_slug}",
                    "code": e.code,
                },
                status=404,
            )

        return JsonResponse(
            {
                "slug": agent.slug,
                "name": agent.name,
                "description": agent.description,
                "parameters": [param.as_dict() for param in agent.parameters or []],
            }
        )

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
            agent = self._get_agent()
        except AgentNotFound as e:
            return JsonResponse(
                {
                    "error": f"Agent not found: {self.agent_slug}",
                    "code": e.code,
                },
                status=404,
            )

        permission = getattr(agent, "permission", None) or AllowAny()
        if not permission.has_permission(request, self.agent_slug):
            return JsonResponse(
                {
                    "error": permission.get_permission_denied_message(
                        request, self.agent_slug
                    ),
                    "code": "permission_denied",
                },
                status=403,
            )

        result = self._execute_agent(agent, data["arguments"])

        return JsonResponse({"status": "completed", "data": result})

    def _get_agent(self) -> "Agent":
        from . import registry

        try:
            return registry.get(self.agent_slug)()
        except KeyError as e:
            raise AgentNotFound from e

    def _execute_agent(self, agent: "Agent", arguments: dict[str, Any]) -> Any:
        return agent.execute(**arguments)
