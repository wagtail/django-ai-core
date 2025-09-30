from abc import ABC, abstractmethod

from django.contrib.auth.models import AnonymousUser
from django.http import HttpRequest


class BasePermission(ABC):
    """
    Base class for implementing custom permission checks on agents.

    Subclass this to create your own permission logic for controlling
    who can execute agents in your application.
    """

    @abstractmethod
    def has_permission(self, request: HttpRequest, agent_slug: str, **kwargs) -> bool:
        """
        Check if the request has permission to execute the agent.

        Args:
            request: The Django HTTP request object
            agent_slug: The slug of the agent being executed
            **kwargs: Additional context that may be useful for permission checks

        Returns:
            bool: True if permission is granted, False otherwise
        """
        pass

    def get_permission_denied_message(
        self, request: HttpRequest, agent_slug: str
    ) -> str:
        """
        Return a custom message when permission is denied.

        Override this method to provide specific error messages.

        Args:
            request: The Django HTTP request object
            agent_slug: The slug of the agent being executed

        Returns:
            str: The permission denied message
        """
        return f"You do not have permission to execute agent '{agent_slug}'"


class AllowAny(BasePermission):
    """
    Permission that allows all requests.

    This is the default behavior if no permission is specified.
    """

    def has_permission(self, request: HttpRequest, agent_slug: str, **kwargs) -> bool:
        return True


class IsAuthenticated(BasePermission):
    """
    Permission that requires the user to be authenticated.
    """

    def has_permission(self, request: HttpRequest, agent_slug: str, **kwargs) -> bool:
        return bool(request.user and request.user.is_authenticated)

    def get_permission_denied_message(
        self, request: HttpRequest, agent_slug: str
    ) -> str:
        return "Authentication required to execute this agent"


class DjangoPermission(BasePermission):
    """
    Permission that uses Django's built-in permission system.

    This checks for a specific Django permission before allowing agent execution.
    The permission should be in the format 'app_label.permission_codename'.

    Args:
        permission: The Django permission string (e.g., 'myapp.can_use_agent')
    """

    def __init__(self, permission: str):
        if "." not in permission:
            raise ValueError(
                "Permission must be in format 'app_label.permission_codename', "
                f"got '{permission}'"
            )
        self.permission = permission

    def has_permission(self, request: HttpRequest, agent_slug: str, **kwargs) -> bool:
        if not request.user or isinstance(request.user, AnonymousUser):
            return False
        return request.user.has_perm(self.permission)

    def get_permission_denied_message(
        self, request: HttpRequest, agent_slug: str
    ) -> str:
        return f"You do not have the required permission '{self.permission}' to execute agent '{agent_slug}'"


class CompositePermission(BasePermission):
    """
    Permission that combines multiple permissions

    Args:
        permissions: List of permissions
        require_all: If True, all permissions must pass. If False, any permission can pass.
    """

    def __init__(self, permissions: list[BasePermission], require_all: bool = True):
        if not permissions:
            raise ValueError("At least one permission must be provided")
        self.permissions = permissions
        self.require_all = require_all

    def has_permission(self, request: HttpRequest, agent_slug: str, **kwargs) -> bool:
        if self.require_all:
            return all(
                permission.has_permission(request, agent_slug, **kwargs)
                for permission in self.permissions
            )
        else:
            return any(
                permission.has_permission(request, agent_slug, **kwargs)
                for permission in self.permissions
            )

    def get_permission_denied_message(
        self, request: HttpRequest, agent_slug: str
    ) -> str:
        for permission in self.permissions:
            if not permission.has_permission(request, agent_slug):
                return permission.get_permission_denied_message(request, agent_slug)
        return f"You do not have permission to execute agent '{agent_slug}'"
