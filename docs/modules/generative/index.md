# Generative

The Generative module is the provider-agnostic layer for text completions and
embeddings. It supersedes the older [`LLMService`](legacy-llm-service.md): it
adds settings-driven role configuration, a stable provider interface you can
subclass, and a semantic exception hierarchy so your app never sees raw vendor
SDK errors.

!!! info "Replaces `LLMService`"

    `django_ai_core.llm.LLMService` is deprecated. New code should use
    `GenerativeService` or `resolve_generative_provider` (below). See
    [Legacy: LLMService](legacy-llm-service.md) if you are migrating.

## Choosing a layer

The module gives you three ways in, layered from most convenient to most
control. Most apps only ever need the service layer.

| Layer | Use it when | Entry point | What you give up |
| --- | --- | --- | --- |
| **Service** | You just want completions or embeddings — text in, text out — and want to swap the model/vendor from settings without touching code. The common case. | `GenerativeService.for_role("chat")` | Only the generic `completion` / `stream` surface; no backend-specific methods. |
| **Direct provider** | You need methods the service does not expose — a custom provider's domain methods (rerank, classify, structured output), or the provider object itself — while still resolving config from settings. | `resolve_generative_provider("chat")` | The thin service wrapper (roles, uniform surface); you hold the provider directly. |
| **Custom provider** | No shipped provider fits — you are integrating a backend that is not covered, or adding domain-specific methods on top of one. Authoring, not just consuming. | Subclass `GenerativeProvider` (in `django_ai_core.generative`) or `EmbeddingProvider` (in `django_ai_core.embedding`) — see [Writing a custom provider](custom-providers.md). | Nothing — but you now own the backend integration and its tests. |

Rule of thumb: **start at the service layer.** Drop to a direct provider only
when you need a method it does not forward, and write a custom provider only when
no shipped provider covers your backend.

All three read model config from settings **roles**. If you need a model chosen
at runtime instead, see [Ad-hoc models without a role](#ad-hoc-models-without-a-role).

## Installation

`any-llm-sdk` ships as a core dependency, but talking to a specific vendor needs
that vendor's client SDK. Install it via the corresponding
[any-llm](https://github.com/mozilla-ai/any-llm) extra:

```bash
pip install "any-llm-sdk[anthropic]"   # Anthropic
pip install "any-llm-sdk[openai]"      # OpenAI
pip install "any-llm-sdk[all]"         # every supported provider
```

any-llm supports many providers — see
[its installation docs](https://docs.mozilla.ai/quickstart#installation) for the
full list of extras. The extra only installs the vendor *client*. Error handling
is provider-agnostic either way (see [Exceptions](#exceptions)).

## Configuration

Roles map a name your code refers to (e.g. `"chat"`, `"summarise"`) onto a
concrete provider and its parameters. Configure them in your Django settings
under `AI_CORE`:

```python
AI_CORE = {
    "GENERATIVE_MODELS": {
        "chat": {
            "provider": "django_ai_core.generative.providers.anyllm.AnyLLMProvider",
            "params": {"provider": "anthropic", "model": "claude-haiku-4-5"},
        },
    },
    "EMBEDDING_MODELS": {
        "default": {
            "provider": "django_ai_core.embedding.providers.anyllm.AnyLLMEmbeddingProvider",
            "params": {"provider": "openai", "model": "text-embedding-3-small"},
        },
    },
}
```

`params` are passed to the provider constructor. For `AnyLLMProvider` that means
`provider` (the any-llm vendor id) and `model`, plus any extra client kwargs.

!!! note "Embeddings: provider ships, index consumption is a TODO"

    `EMBEDDING_MODELS`, `resolve_embedding_provider`, and the shipped
    `AnyLLMEmbeddingProvider` all work today. What is not yet wired is
    *consumption inside the library*: the index module still embeds via the
    deprecated `LLMService` rather than an `EmbeddingProvider`. You can also
    point a role at your own
    [custom `EmbeddingProvider`](custom-providers.md#embedding-providers).

Misconfiguration is surfaced eagerly as Django's `ImproperlyConfigured` at
resolution time (missing role, missing `provider` key, unimportable path, wrong
base class, bad constructor params).

## Service layer

Highest-level entry point, via `GenerativeService`. Resolve a role from settings
and call it generically:

```python
from django_ai_core.generative import GenerativeService

service = GenerativeService.for_role("chat")

text = service.completion("Summarise the plot of Hamlet.")

async_text = await service.acompletion("...")

for delta in service.stream("Tell me a story."):
    print(delta, end="")

async for delta in service.astream("..."):
    print(delta, end="")
```

`stream` / `astream` emit a single lifecycle log line
(`django_ai_core.generative.service`, INFO) when the stream ends — on normal
completion, client disconnect (`cancelled`), or error (`errored`) — carrying
`state`, `role`, `model`, chunk/char/word counts, elapsed time, and best-effort
token usage. Usage counts are populated on a clean finish and `None` on cancel.

Construct directly for tests or per-tenant config:

```python
GenerativeService(provider=my_provider, role="chat")
```

## Direct provider

When you need the provider instance itself — e.g. to call a custom provider's
domain methods — resolve it without the service wrapper:

```python
from django_ai_core.generative import resolve_generative_provider

provider = resolve_generative_provider("chat")
provider.completion("...")

# narrow the return type when you rely on a subclass's methods:
provider = resolve_generative_provider("rerank", expect=MyProvider)
provider.rerank(docs)
```

`resolve_embedding_provider` is the embedding counterpart. Each call returns a
**fresh** instance — no caching at this layer.

To build a provider of your own, see
[Writing a custom provider](custom-providers.md).

## Ad-hoc models without a role

Roles are the intended way to configure models — stable, named, swappable from
settings. But some apps need to pick a model at **runtime** that was never
pre-configured: a "choose your model" dropdown, a per-request override, a tenant
that brings its own model id. For that, skip roles entirely and construct a
provider inline with whatever model you like:

```python
from django_ai_core.generative import GenerativeService
from django_ai_core.generative.providers.anyllm import AnyLLMProvider

# model chosen at runtime — e.g. from a form the user submitted
provider = AnyLLMProvider(provider="anthropic", model=user_selected_model)

# use it directly...
text = provider.completion("...")

# ...or wrap it in a service for the uniform surface:
service = GenerativeService(provider=provider)
text = service.completion("...")
```

Nothing here touches `AI_CORE` settings — the provider is fully described by its
constructor arguments, so any valid `provider` / `model` combination works
without a matching role.

!!! note "Roles vs. inline"

    Prefer a role when the set of models is known and fixed — it keeps model
    choice out of code and swappable from settings. Reach for inline
    construction only when the model genuinely is not known ahead of time.
    There is deliberately **no** model-override argument on
    `resolve_generative_provider`: overriding a role's configured params at call
    time would blur the "config lives in settings" boundary, and inline
    construction already covers the runtime case cleanly.

## The shipped AnyLLM provider

`AnyLLMProvider` is batteries-included: it talks to every any-llm vendor and
translates failures into the [semantic exceptions](#exceptions) below, so you
write no provider code of your own. See
[The AnyLLM provider](anyllm-provider.md) for configuration and error mapping.

## Exceptions

All provider failures raise a subclass of `AICoreProviderError`. Names describe
*semantics*, not transport — a provider maps its own failures into this
vocabulary from the outside, so your handling code is provider-agnostic.

```
AICoreProviderError                 # root: any provider failure (generative OR embedding)
├── ProviderConfigurationError      # runtime misconfig: bad/missing key, unknown
│                                    # model, unsupported param. Caller error.
└── ProviderRequestError            # a real request attempt failed
    ├── ProviderTimeoutError        # request deadline exceeded
    ├── ProviderUnavailableError    # backend reached but refusing: overloaded / at capacity
    ├── ProviderRateLimitError      # throttled / quota exhausted
    ├── ProviderResponseError       # response received but unusable: empty text,
    │                                # malformed embedding, safety-filtered output
    └── ProviderUnexpectedError     # request failed, cause unclassified: transport
                                     # failure, 5xx, or an error we can't map
```

Catch broad or narrow:

```python
from django_ai_core.generative import (
    AICoreProviderError,
    ProviderRequestError,
    ProviderRateLimitError,
)

try:
    service.completion("...")
except ProviderRateLimitError:
    ...                     # one specific cause
except ProviderRequestError:
    ...                     # any failed request attempt (not config errors)
except AICoreProviderError:
    ...                     # anything, including configuration errors
```

!!! note "Two kinds of configuration error"

    Settings-wiring problems found at resolution time raise Django's
    `ImproperlyConfigured`. `ProviderConfigurationError` is for misconfiguration
    discovered later, at call time (e.g. the provider rejects the credentials).

## Roadmap

Routing every AI request through a single resolved layer gives Django AI Core one
place to add cross-cutting infrastructure. Planned:

- **Request logging** — a record of every completion / embedding call.
  Streaming already emits a per-stream lifecycle log line (state, model, counts,
  best-effort usage); a durable audit record is the next step.
- **Cost calculation** — token and spend accounting per role.
- **Rate limiting** — throttle usage per role or tenant.

These land in the Generative layer, so code that already resolves providers
through roles picks them up without changes.
