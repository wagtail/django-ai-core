# The AnyLLM provider

`AnyLLMProvider` is the batteries-included provider shipped with Django AI Core.
It lives in the library, so you reference it by dotted path and write no provider
code of your own:

```
django_ai_core.generative.providers.anyllm.AnyLLMProvider
```

It is backed by [`any-llm`](https://mozilla-ai.github.io/any-llm/), giving a
single interface across OpenAI, Anthropic, Gemini, Amazon Bedrock, self-hosted
models via Ollama, and [more](https://mozilla-ai.github.io/any-llm/providers/).

## Configuration

Point a role at it and pass the vendor and model through `params`:

```python
AI_CORE = {
    "GENERATIVE_MODELS": {
        "chat": {
            "provider": "django_ai_core.generative.providers.anyllm.AnyLLMProvider",
            "params": {"provider": "anthropic", "model": "claude-haiku-4-5"},
        },
    },
}
```

`params` are passed to the constructor: `provider` (the any-llm vendor id) and
`model` are required, plus any extra client kwargs. Talking to a given vendor
also needs that vendor's client SDK — install it via the matching any-llm extra
(e.g. `any-llm-sdk[anthropic]`, see [Installation](index.md#installation)).

## Response normalisation

any-llm normalises every vendor's response to the OpenAI-shaped `ChatCompletion`
/ `ChatCompletionChunk`, so the provider extracts text uniformly regardless of
backend — `.choices[0].message.content` for a completion,
`.choices[0].delta.content` for a stream chunk. An empty-but-successful response
(a 200 with no content) is surfaced as
[`ProviderResponseError`](index.md#exceptions).

## Error translation

Failures are translated into the [semantic exception hierarchy](index.md#exceptions).
Consumers never see a raw vendor error or an `AnyLLMError`.

!!! warning "`ANY_LLM_UNIFIED_EXCEPTIONS`"

    Error translation relies on any-llm's unified exceptions. any-llm only
    converts raw vendor SDK errors into its own `AnyLLMError` hierarchy when the
    `ANY_LLM_UNIFIED_EXCEPTIONS` environment variable is enabled; otherwise it
    re-raises the raw vendor error with a deprecation warning.

    `AnyLLMProvider` enables it on import with
    `os.environ.setdefault("ANY_LLM_UNIFIED_EXCEPTIONS", "1")` — so it works out
    of the box and an explicit choice you set yourself is left untouched. **If
    you set `ANY_LLM_UNIFIED_EXCEPTIONS=0` (or any falsey value) in the
    environment, translation degrades**: unrecognised vendor errors fall through
    to the generic `ProviderResponseError` instead of the precise semantic type.
    Leave it unset, or set it to `1`.

The unified any-llm error → semantic exception mapping:

| any-llm exception | Semantic exception |
| --- | --- |
| `RateLimitError` | `ProviderRateLimitError` |
| `GatewayTimeoutError` | `ProviderTimeoutError` |
| `UpstreamProviderError` | `ProviderUnavailableError` |
| `AuthenticationError`, `MissingApiKeyError`, `UnsupportedProviderError`, `UnsupportedParameterError`, `InvalidRequestError`, `ModelNotFoundError`, `ContextLengthExceededError`, `InsufficientFundsError` | `ProviderConfigurationError` |
| anything else (generic `ProviderError`, content / finish-reason errors, non-any-llm exceptions) | `ProviderResponseError` |

Refining the mapping is a one-line edit to `_EXCEPTION_MAP` in the provider — no
vendor SDK code involved.
