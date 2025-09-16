# Django AI Core Documentation

Django AI Core provides a set of tools for implementing AI-powered features in to your Django sites. It currently includes tools for indexing and searching content in vector databases and building/running AI agents.

## Quick Start

### Installation

```bash
pip install django-ai-core
```

### Basic Setup

Add `django_ai_core` to `INSTALLED_APPS`, along with any `contrib` modules you need:

-   `django_ai_core.contrib.index` - vector indexing and searching across your data
-   `django_ai_core.contrib.agents` - register AI agents that can do some AI tasks

```python
INSTALLED_APPS = [
    'django_ai_core',
]
```

Run migrations:

```bash
python manage.py migrate
```

### Read More

-   [Core](modules/core/) - on using the core module to access low-level AI tooling
-   [Index Module](modules/index) - on indexing your data for similarity search and for powering RAG applications
-   [Agent Module](modules/agents/) - on creating AI tools that can be triggered from other parts of your app
