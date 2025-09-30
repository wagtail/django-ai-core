import json

import pytest
from django.contrib.auth.models import AnonymousUser, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.test import RequestFactory

from django_ai_core.contrib.agents import (
    Agent,
    AgentParameter,
    registry,
)
from django_ai_core.contrib.agents.permissions import (
    CompositePermission,
    DjangoPermission,
    IsAuthenticated,
)
from django_ai_core.contrib.agents.views import AgentExecutionView


class AnonymousAgent(Agent):
    """Agent without permission requirements."""

    slug = "anonymous-agent"
    description = "A agent accessible to the public"
    parameters = [
        AgentParameter(
            name="message",
            type=str,
            description="A message to echo",
        ),
    ]

    def execute(self, message: str) -> str:
        return f"Public: {message}"


class AuthenticatedAgent(Agent):
    """Agent requiring authentication."""

    slug = "authenticated-agent"
    description = "An agent requiring authentication"
    permission = IsAuthenticated()
    parameters = [
        AgentParameter(
            name="message",
            type=str,
            description="A message to echo",
        ),
    ]

    def execute(self, message: str) -> str:
        return f"Authenticated: {message}"


class PermissionRequiredAgent(Agent):
    """Agent requiring specific Django permission."""

    slug = "permission-agent"
    description = "An agent requiring a specific permission"
    permission = DjangoPermission("auth.view_user")
    parameters = [
        AgentParameter(
            name="message",
            type=str,
            description="A message to echo",
        ),
    ]

    def execute(self, message: str) -> str:
        return f"Permission: {message}"


class CompositePermissionAgent(Agent):
    """Agent with composite permission requirements."""

    slug = "composite-agent"
    description = "An agent with composite permissions"
    permission = CompositePermission(
        [IsAuthenticated(), DjangoPermission("auth.change_user")],
        require_all=True,
    )
    parameters = [
        AgentParameter(
            name="message",
            type=str,
            description="A message to echo",
        ),
    ]

    def execute(self, message: str) -> str:
        return f"Composite: {message}"


@pytest.fixture
def register_test_agents():
    """Register test agents for the test session."""
    registry.register()(AnonymousAgent)
    registry.register()(AuthenticatedAgent)
    registry.register()(PermissionRequiredAgent)
    registry.register()(CompositePermissionAgent)


@pytest.fixture
def factory():
    """Request factory fixture."""
    return RequestFactory()


@pytest.mark.django_db
class TestAgentPermission:
    """Tests for agent permission checking."""

    def test_anonymous_agent_allows_anonymous(self, factory, register_test_agents):
        """Test that anonymous agents allow anonymous users."""
        view = AgentExecutionView()
        view.agent_slug = "anonymous-agent"

        request = factory.post(
            "/ai/anonymous-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = AnonymousUser()

        response = view.post(request)
        assert response.status_code == 200

        data = json.loads(response.content)
        assert data["status"] == "completed"
        assert data["data"] == "Public: hello"

    def test_anonymous_agent_allows_authenticated(self, factory, register_test_agents):
        """Test that anonymous agents allow authenticated users."""
        user = User.objects.create_user(username="testuser", password="test123")

        view = AgentExecutionView()
        view.agent_slug = "anonymous-agent"

        request = factory.post(
            "/ai/anonymous-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 200

        data = json.loads(response.content)
        assert data["status"] == "completed"
        assert data["data"] == "Public: hello"

    def test_authenticated_agent_denies_anonymous(self, factory, register_test_agents):
        """Test that authenticated agents deny anonymous users."""
        view = AgentExecutionView()
        view.agent_slug = "authenticated-agent"

        request = factory.post(
            "/ai/authenticated-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = AnonymousUser()

        response = view.post(request)
        assert response.status_code == 403

        data = json.loads(response.content)
        assert data["code"] == "permission_denied"
        assert "Authentication required" in data["error"]

    def test_authenticated_agent_allows_authenticated(
        self, factory, register_test_agents
    ):
        """Test that authenticated agents allow authenticated users."""
        user = User.objects.create_user(username="testuser", password="test123")

        view = AgentExecutionView()
        view.agent_slug = "authenticated-agent"

        request = factory.post(
            "/ai/authenticated-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 200

        data = json.loads(response.content)
        assert data["status"] == "completed"
        assert data["data"] == "Authenticated: hello"

    def test_permission_agent_denies_without_permission(
        self, factory, register_test_agents
    ):
        """Test that permission agents deny users without the required permission."""
        user = User.objects.create_user(username="testuser", password="test123")

        view = AgentExecutionView()
        view.agent_slug = "permission-agent"

        request = factory.post(
            "/ai/permission-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 403

        data = json.loads(response.content)
        assert data["code"] == "permission_denied"
        assert "auth.view_user" in data["error"]

    def test_permission_agent_allows_with_permission(
        self, factory, register_test_agents
    ):
        """Test that permission agents allow users with the required permission."""
        user = User.objects.create_user(username="testuser", password="test123")

        # Grant the required permission
        content_type = ContentType.objects.get_for_model(User)
        permission = Permission.objects.get(
            codename="view_user",
            content_type=content_type,
        )
        user.user_permissions.add(permission)

        view = AgentExecutionView()
        view.agent_slug = "permission-agent"

        request = factory.post(
            "/ai/permission-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 200

        data = json.loads(response.content)
        assert data["status"] == "completed"
        assert data["data"] == "Permission: hello"

    def test_composite_permission_requires_all(self, factory, register_test_agents):
        """Test that composite permissions require all permissions to pass."""
        user = User.objects.create_user(username="testuser", password="test123")

        view = AgentExecutionView()
        view.agent_slug = "composite-agent"

        request = factory.post(
            "/ai/composite-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 403

        content_type = ContentType.objects.get_for_model(User)
        permission = Permission.objects.get(
            codename="change_user",
            content_type=content_type,
        )
        user.user_permissions.add(permission)

        # Refresh user from database to load new permissions
        user = User.objects.get(pk=user.pk)

        # Now should succeed
        request = factory.post(
            "/ai/composite-agent/",
            data=json.dumps({"arguments": {"message": "hello"}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 200

        data = json.loads(response.content)
        assert data["status"] == "completed"
        assert data["data"] == "Composite: hello"

    def test_nonexistent_agent_returns_404(self, factory, register_test_agents):
        """Test that requesting a nonexistent agent returns 404."""
        user = User.objects.create_user(username="testuser", password="test123")

        view = AgentExecutionView()
        view.agent_slug = "nonexistent-agent"

        request = factory.post(
            "/ai/nonexistent-agent/",
            data=json.dumps({"arguments": {}}),
            content_type="application/json",
        )
        request.user = user

        response = view.post(request)
        assert response.status_code == 404

        data = json.loads(response.content)
        assert data["code"] == "agent_not_found"
