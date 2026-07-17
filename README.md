# Django AI Core

**Developer-focused package for implementing AI tooling into Django sites.**

Django AI Core provides core functionality for interacting with LLMs in your Django project, as well as optional modules for building AI-powered apps.

## Installation

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

See the [Generative module docs](docs/modules/generative.md) for provider configuration and usage.

## TODO

-   Core:
    -   Logging for LLM requests
    -   Async/streaming support
-   Agents:
    -   Rate limiting
    -   Streaming responses
-   Index:
    -   Test different backends
    -   Docs
    -   Higher-level API?
-   Chat:
    -   Build Chat/RAG module with chat UI
-   MCP:
    -   Build MCP 'function' registry
