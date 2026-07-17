# Legacy: LLMService

!!! warning "Deprecated"

    `LLMService` is deprecated and will be removed in a future release. It emits
    a `DeprecationWarning` at runtime. Use the [Generative module](index.md)
    instead — `GenerativeService.for_role(...)` for settings-driven config, or
    `resolve_generative_provider(...)` to use a provider directly.

    This page documents the old API for existing code only. New code should not
    use it.

`LLMService` is a light wrapper around [`any-llm`](https://mozilla-ai.github.io/any-llm/)
for requesting completions or embeddings from many AI providers, including
OpenAI, Anthropic, Gemini, Amazon Bedrock and self-hosted models through Ollama.
See the [full list of supported providers](https://mozilla-ai.github.io/any-llm/providers/).

To use the `LLMService`:

```python
from django_ai_core.llm import LLMService

service = LLMService.create(
    provider="openai",
    model="gpt-4o",
)
```

You can also instantiate `LLMService` with your own client instance:

```python
from any_llm import AnyLLM

client = AnyLLM.create(provider="openai", model="gpt-4o")

service = LLMService(client=client, model="gpt-4o")
```

## Completions

```python
response = service.completion(
    "What is the airspeed velocity of an unladen swallow?"
)
```

## Embeddings

```python
response = service.embedding(
    "What's the speed on that bird when it's not hauling stuff?"
)
```

All keyword arguments are passed to the underlying `any-llm`
[`completion`](https://mozilla-ai.github.io/any-llm/api/completion/) and
[`embedding`](https://mozilla-ai.github.io/any-llm/api/embedding/) APIs.
`any-llm` normalises responses from all providers to OpenAI's API schema.

## Migrating to the Generative module

| `LLMService` | Generative equivalent |
| --- | --- |
| `LLMService.create(provider=..., model=...)` | Configure a role in `AI_CORE['GENERATIVE_MODELS']`, then `GenerativeService.for_role(name)` |
| `service.completion(messages)` | `service.completion(prompt)` (see [Service layer](index.md#service-layer)) |
| `service.embedding(inputs)` | Configure `AI_CORE['EMBEDDING_MODELS']`, then `resolve_embedding_provider(name)` |

The Generative layer moves provider choice into settings, so swapping vendor or
model no longer touches code, and it translates failures into the
[semantic exception hierarchy](index.md#exceptions).
