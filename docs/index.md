# Django AI Core Documentation

Django AI Core provides a set of tools for implementing AI-powered features in to your Django sites. It currently includes tools for indexing and searching content in vector databases and building/running AI agents.

## Quick Start

### Installation

```bash
pip install django-ai-core
```

Talking to a specific LLM vendor needs that vendor's client SDK. Install it via the corresponding [any-llm](https://github.com/mozilla-ai/any-llm) extra:

```bash
pip install "any-llm-sdk[anthropic]"   # Anthropic
pip install "any-llm-sdk[openai]"      # OpenAI
pip install "any-llm-sdk[all]"         # every supported provider
```

any-llm supports many providers — see [its installation docs](https://docs.mozilla.ai/quickstart#installation) for the full list of extras.

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

-   [Generative](modules/generative/) - completions and embeddings via configurable providers (the completion/embedding entry point; supersedes the deprecated `LLMService`)
-   [Index Module](modules/index) - on indexing your data for similarity search and for powering RAG applications
-   [Agent Module](modules/agents/) - on creating AI tools that can be triggered from other parts of your app
