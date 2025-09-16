from django.urls import path

from . import registry
from .views import AgentExecutionView


def agent_urls() -> list:
    """
    Generate URL patterns for all registered agents.

    Returns:
        List of URL patterns

    Example:
        # In your main urls.py
        from django_ai_core.agent import generate_agent_urls

        urlpatterns = [
            # ... your other URLs
            path('ai/', include(generate_agent_urls())),
        ]
    """
    urlpatterns = []

    for agent_slug in registry.list().keys():
        urlpatterns.append(
            path(
                f"{agent_slug}/",
                AgentExecutionView.as_view(agent_slug=agent_slug),
                name=f"agent_{agent_slug}",
            )
        )

    return urlpatterns
