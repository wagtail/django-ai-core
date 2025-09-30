# Core Infrastructure

The Core module provides shared infrastructure that powers all other modules in Django AI Core. It currently offers simple wrapper services for completion/embedding.

!!! note "Future plans"

    By routing all AI requests through these core service we have a single place to start tracking any queries your app makes, supporting future planned features like:

    - Request logging
    - Cost calculations
    - Rate limiting

## LLM Service

Django AI Core provides an `LLMService` which can be used for requesting completions or embeddings from many AI providers.

It uses the [`any-llm`](https://mozilla-ai.github.io/any-llm/) package to provide a unified interface across multiple LLM providers including OpenAI, Anthropic, Gemini, Amazon Bedrock and self-hosted models through Ollama. See the [full list of supported providers](https://mozilla-ai.github.io/any-llm/providers/).

To use the `LLMService`:

```python

from django_ai_core.llm import LLMService

service = LLMService.create(
    provider="openai",
    model="gpt-4o"
)
```

You can also alternatively instantiate `LLMService` with your own client instance:

```python
from any_llm import AnyLLM

client = AnyLLM.create(
provider="openai",
model="gpt-4o"
)

service = LLMService(client=client)
```

# Completions

```python
response = service.completion(
    "What is the airspeed velocity of an unladen swallow?"
)
```

# Embeddings

```python
response = service.embedding(
    "What's the speed on that bird when it's not hauling stuff?"
)
```

All keyword arguments are passed to the underlying `any-llm` [`completion`](https://mozilla-ai.github.io/any-llm/api/completion/) and [`embedding`](https://mozilla-ai.github.io/any-llm/api/embedding/) APIs.

`any-llm` normalises responses from all providers to OpenAI's API schema.
