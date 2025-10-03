# Agent Permissions

The Agent Permissions system allows you to control who can execute specific agents in your application. You can use the built-in permissions or create your own custom permission logic.

## Overview

By default, agents are accessible to all users. To restrict access, you can add a `permission` to your agent class.

This permission will be checked before the agent can be accessed through the automatically generated view. The permission is not checked when executing `execute` on the agent manually in code.

## Built-in Permissions

### AllowAny

Allows all requests (this is the default behavior if no permission is specified).

```python
from django_ai_core.contrib.agents import Agent
from django_ai_core.contrib.agents.permissions import AllowAny

class MyAgent(Agent):
    slug = "my-agent"
    name = "My Agent"
    permission = AllowAny()

    def execute(self):
        return "Hello!"
```

### IsAuthenticated

Requires the user to be authenticated.

```python
from django_ai_core.contrib.agents import Agent
from django_ai_core.contrib.agents.permissions import IsAuthenticated

class SecureAgent(Agent):
    slug = "secure-agent"
    name = 'Secure Agent"
    description = "Agent that requires authentication"
    permission = IsAuthenticated()
    parameters = []

    def execute(self):
        return "You are authenticated!"
```

### DjangoPermission

Uses Django's built-in permission system to check for a specific permission. The permission should be in the format `'app_label.permission_codename'`.

```python
from django_ai_core.contrib.agents import Agent
from django_ai_core.contrib.agents.permissions import DjangoPermission

class AdminAgent(Agent):
    slug = "admin-agent"
    name = "Admin Agent"
    description = "Agent requiring admin permission"
    permission = DjangoPermission("myapp.can_use_admin_agent")
    parameters = []

    def execute(self):
        return "Admin access granted!"
```

### CompositePermission

Combines multiple permissions with AND or OR logic.

```python
from django_ai_core.contrib.agents import Agent
from django_ai_core.contrib.agents.permissions import (
    CompositePermission,
    IsAuthenticated,
    DjangoPermission,
)

class SuperSecureAgent(Agent):
    slug = "super-secure-agent"
    name = "Super Secure Agent"
    description = "Agent with multiple permission requirements"
    permission = CompositePermission(
        [
            IsAuthenticated(),
            DjangoPermission("myapp.can_use_secure_features")
        ],
        require_all=True  # All permissions must pass (logical AND)
    )
    parameters = []

    def execute(self):
        return "Super secure data!"

class FlexibleAgent(Agent):
    slug = "flexible-agent"
    name = "Flexible Agent"
    description = "Agent with flexible permission requirements"
    permission = CompositePermission(
        [
            DjangoPermission("myapp.admin_access"),
            DjangoPermission("myapp.power_user")
        ],
        require_all=False  # Any permission can pass (logical OR)
    )
    parameters = []

    def execute(self):
        return "Flexible access!"
```

## Creating Custom Permissions

You can create your own permissions by subclassing `BasePermission`:

```python
from django_ai_core.contrib.agents import Agent
from django_ai_core.contrib.agents.permissions import BasePermission

class IPAllowlist(BasePermission):
    """Only allow requests from specific IP addresses."""

    def __init__(self, allowed_ips):
        self.allowed_ips = allowed_ips

    def has_permission(self, request, agent_slug, **kwargs):
        client_ip = request.META.get('REMOTE_ADDR')
        return client_ip in self.allowed_ips

    def get_permission_denied_message(self, request, agent_slug):
        return f"Your IP address is not authorized to execute '{agent_slug}'"

class RestrictedAgent(Agent):
    slug = "restricted-agent"
    name = "Restricted Agent"
    description = "Agent only accessible from specific IPs"
    permission = IPAllowlist(['192.168.1.100', '10.0.0.5'])
    parameters = []

    def execute(self):
        return "Access granted from allowlisted IP!"
```

## Permission Denied Responses

When a permission check fails, the agent execution view returns a 403 Forbidden response:

```json
{
    "error": "You do not have permission to execute agent 'my-agent'",
    "code": "permission_denied"
}
```

You can customize this message by overriding the `get_permission_denied_message()` method in your permission class.
