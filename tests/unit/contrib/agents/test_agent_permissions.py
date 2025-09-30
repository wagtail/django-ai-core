from unittest.mock import Mock

import pytest
from django.contrib.auth.models import AnonymousUser, Permission, User
from django.contrib.contenttypes.models import ContentType
from django.http import HttpRequest

from django_ai_core.contrib.agents.permissions import (
    AllowAny,
    BasePermission,
    CompositePermission,
    DjangoPermission,
    IsAuthenticated,
)


class TestPermission:
    """Test the base Permission class."""

    def test_abstract_method(self):
        """Test that Permission cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePermission()

    def test_custom_permission(self):
        """Test creating a custom permission."""

        class IsAdminUser(BasePermission):
            def has_permission(self, request, agent_slug, **kwargs):
                return request.user.username == "admin"

        permission = IsAdminUser()
        valid_request = Mock(spec=HttpRequest)
        valid_request.user = Mock(username="admin")

        invalid_request = Mock(spec=HttpRequest)
        invalid_request.user = Mock(username="guest")

        assert permission.has_permission(valid_request, "test-agent") is True
        assert permission.has_permission(invalid_request, "test-agent") is False

    def test_custom_permission_denied_message(self):
        """Test custom permission denied messages."""

        class CustomMessage(BasePermission):
            def has_permission(self, request, agent_slug, **kwargs):
                return False

            def get_permission_denied_message(self, request, agent_slug):
                return f"Custom message for {agent_slug}"

        permission = CustomMessage()
        request = Mock(spec=HttpRequest)

        message = permission.get_permission_denied_message(request, "my-agent")
        assert message == "Custom message for my-agent"


class TestAllowAny:
    """Test the AllowAny permission."""

    def test_allows_authenticated(self):
        """Test that AllowAny allows authenticated users."""
        permission = AllowAny()
        request = Mock(spec=HttpRequest)

        request.user = Mock(is_authenticated=True)
        assert permission.has_permission(request, "test-agent") is True

    def test_allows_anonymous(self):
        """Test that AllowAny allows anonymous users."""
        permission = AllowAny()
        request = Mock(spec=HttpRequest)

        request.user = AnonymousUser()
        assert permission.has_permission(request, "test-agent") is True

    def test_allows_none(self):
        """Test that AllowAny allows a 'None' user."""
        permission = AllowAny()
        request = Mock(spec=HttpRequest)

        request.user = None
        assert permission.has_permission(request, "test-agent") is True


class TestIsAuthenticated:
    """Test the IsAuthenticated permission."""

    def test_allows_authenticated_user(self):
        """Test that authenticated users are allowed."""
        permission = IsAuthenticated()
        request = Mock(spec=HttpRequest)
        request.user = Mock(is_authenticated=True)

        assert permission.has_permission(request, "test-agent") is True

    def test_denies_anonymous_user(self):
        """Test that anonymous users are denied."""
        permission = IsAuthenticated()
        request = Mock(spec=HttpRequest)
        request.user = AnonymousUser()

        assert permission.has_permission(request, "test-agent") is False

    def test_denies_no_user(self):
        """Test that requests with no user are denied."""
        permission = IsAuthenticated()
        request = Mock(spec=HttpRequest)
        request.user = None

        assert permission.has_permission(request, "test-agent") is False

    def test_custom_denied_message(self):
        """Test the custom denied message."""
        permission = IsAuthenticated()
        request = Mock(spec=HttpRequest)
        request.user = AnonymousUser()

        message = permission.get_permission_denied_message(request, "test-agent")
        assert message == "Authentication required to execute this agent"


@pytest.mark.django_db
class TestDjangoPermission:
    """Test the DjangoPermission."""

    def test_invalid_permission_format(self):
        """Test that invalid permission format raises ValueError."""
        with pytest.raises(ValueError, match="must be in format") as excinfo:
            DjangoPermission("invalid_permission")
        assert "app_label.permission_codename" in str(excinfo.value)

    def test_valid_permission_format(self):
        """Test that valid permission format is accepted."""
        permission = DjangoPermission("myapp.can_do_something")
        assert permission.permission == "myapp.can_do_something"

    @pytest.mark.django_db
    def test_allows_user_with_permission(self):
        """Test that users with the required permission are allowed."""
        user = User.objects.create_user(username="testuser", password="test123")

        # Create a content type and permission
        content_type = ContentType.objects.get_for_model(User)
        permission = Permission.objects.create(
            codename="can_use_agent",
            name="Can use agent",
            content_type=content_type,
        )
        user.user_permissions.add(permission)

        permission = DjangoPermission("auth.can_use_agent")
        request = Mock(spec=HttpRequest)
        request.user = user

        assert permission.has_permission(request, "test-agent") is True

    def test_denies_user_without_permission(self):
        """Test that users without the required permission are denied."""
        user = User.objects.create_user(username="testuser", password="test123")

        permission = DjangoPermission("auth.can_use_agent")
        request = Mock(spec=HttpRequest)
        request.user = user

        assert permission.has_permission(request, "test-agent") is False

    def test_denies_anonymous_user(self):
        """Test that anonymous users are always denied."""
        permission = DjangoPermission("auth.can_use_agent")
        request = Mock(spec=HttpRequest)
        request.user = AnonymousUser()

        assert permission.has_permission(request, "test-agent") is False

    def test_custom_denied_message(self):
        """Test the custom denied message includes permission."""
        permission = DjangoPermission("myapp.can_do_something")
        request = Mock(spec=HttpRequest)
        request.user = AnonymousUser()

        message = permission.get_permission_denied_message(request, "test-agent")
        assert "myapp.can_do_something" in message
        assert "test-agent" in message


class TestCompositePermission:
    """Test the CompositePermission."""

    def test_requires_at_least_one_permission(self):
        """Test that at least one permission must be provided."""
        with pytest.raises(ValueError, match="At least one") as excinfo:
            CompositePermission([])
        assert "at least one permission" in str(excinfo.value).lower()

    def test_require_all_with_all_passing(self):
        """Test require_all=True when all permissions pass."""
        permission1 = Mock(spec=BasePermission)
        permission1.has_permission.return_value = True

        permission2 = Mock(spec=BasePermission)
        permission2.has_permission.return_value = True

        composite = CompositePermission([permission1, permission2], require_all=True)
        request = Mock(spec=HttpRequest)

        assert composite.has_permission(request, "test-agent") is True

    def test_require_all_with_one_failing(self):
        """Test require_all=True when one permission fails."""
        permission1 = Mock(spec=BasePermission)
        permission1.has_permission.return_value = True

        permission2 = Mock(spec=BasePermission)
        permission2.has_permission.return_value = False

        composite = CompositePermission([permission1, permission2], require_all=True)
        request = Mock(spec=HttpRequest)

        assert composite.has_permission(request, "test-agent") is False

    def test_require_any_with_one_passing(self):
        """Test require_all=False when at least one permission passes."""
        permission1 = Mock(spec=BasePermission)
        permission1.has_permission.return_value = False

        permission2 = Mock(spec=BasePermission)
        permission2.has_permission.return_value = True

        composite = CompositePermission([permission1, permission2], require_all=False)
        request = Mock(spec=HttpRequest)

        assert composite.has_permission(request, "test-agent") is True

    def test_require_any_with_all_failing(self):
        """Test require_all=False when all permissions fail."""
        permission1 = Mock(spec=BasePermission)
        permission1.has_permission.return_value = False

        permission2 = Mock(spec=BasePermission)
        permission2.has_permission.return_value = False

        composite = CompositePermission([permission1, permission2], require_all=False)
        request = Mock(spec=HttpRequest)

        assert composite.has_permission(request, "test-agent") is False

    def test_returns_first_failing_message(self):
        """Test that the first failing permission's message is returned."""
        permission1 = Mock(spec=BasePermission)
        permission1.has_permission.return_value = False
        permission1.get_permission_denied_message.return_value = (
            "First permission failed"
        )

        permission2 = Mock(spec=BasePermission)
        permission2.has_permission.return_value = False
        permission2.get_permission_denied_message.return_value = (
            "Second permission failed"
        )

        composite = CompositePermission([permission1, permission2])
        request = Mock(spec=HttpRequest)

        message = composite.get_permission_denied_message(request, "test-agent")
        assert message == "First permission failed"

    def test_with_real_permissions_authenticated(self):
        """Test integration with real permissions."""
        composite = CompositePermission(
            [IsAuthenticated(), AllowAny()], require_all=True
        )
        request = Mock(spec=HttpRequest)
        request.user = Mock(is_authenticated=True)

        assert composite.has_permission(request, "test-agent") is True

    def test_with_real_permissions_anonymous(self):
        """Test integration with real permissions."""
        composite = CompositePermission(
            [IsAuthenticated(), AllowAny()], require_all=True
        )
        request = Mock(spec=HttpRequest)
        request.user = AnonymousUser()
        assert composite.has_permission(request, "test-agent") is False

    def test_integration_with_any_logic(self):
        """Test integration with require_all=False logic."""
        composite = CompositePermission(
            [IsAuthenticated(), AllowAny()], require_all=False
        )
        request = Mock(spec=HttpRequest)
        request.user = AnonymousUser()

        assert composite.has_permission(request, "test-agent") is True
