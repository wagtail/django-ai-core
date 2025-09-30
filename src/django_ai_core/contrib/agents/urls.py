from django.urls import path

from . import registry


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

    for agent in registry.list().values():
        urlpatterns.append(
            path(
                f"{agent.slug}/",
                agent.as_view(),
                name=f"agent_{agent.slug}",
            )
        )

    return urlpatterns
