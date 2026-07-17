# Writing a custom provider

Write a custom provider when no shipped provider fits — you are integrating a
backend that [`AnyLLMProvider`](anyllm-provider.md) does not
cover, or you want domain-specific methods (rerank, classify, structured output)
on top of a backend. If you only need generic completions from a supported
vendor, stay on the [service layer](index.md#choosing-a-layer) instead.

## The interfaces

Two abstract classes live in `django_ai_core.generative`, split because the jobs are split:

- `GenerativeProvider` — text completions.
- `EmbeddingProvider` — vector embeddings.

One class may implement both if the backend does both jobs.

### GenerativeProvider

```python
from django_ai_core.generative import GenerativeProvider
from django_ai_core.generative.providers.base import UsageCapture

class MyProvider(GenerativeProvider):
    def __init__(self, *, model: str, **client_kwargs):
        self._client = MyBackend(**client_kwargs)
        self.model = model                          # see "The model attribute"

    def completion(self, prompt, **kwargs):        # required
        ...

    async def acompletion(self, prompt, **kwargs):  # required
        ...

    def stream(                                       # optional
        self, prompt, *, usage_capture: UsageCapture | None = None, **kwargs
    ):
        ...

    async def astream(                                # optional
        self, prompt, *, usage_capture: UsageCapture | None = None, **kwargs
    ):
        ...

    def rerank(self, docs):                         # custom domain method
        ...
```

- **`completion` / `acompletion` are required** (`@abstractmethod`) — a
  generative provider must at minimum produce a completion.
- **`stream` / `astream` are optional.** The base implementation raises
  `NotImplementedError`, so a backend that cannot stream just leaves them alone.
  **Override them whenever the backend does support streaming** — otherwise
  callers of `stream` / `astream` (and `GenerativeService.stream` / `.astream`)
  get a `NotImplementedError` even though the backend could have streamed. See
  [Streaming, usage & cancellation](#streaming-usage-cancellation) for the
  `usage_capture` contract.
- **Custom domain methods** (like `rerank`) go on the subclass, not on
  `GenerativeService`. Reach them via the
  [direct provider](index.md#direct-provider) layer.

#### The `model` attribute

`GenerativeProvider` declares `model: str | None = None`. Set it in `__init__`
so it carries a human-readable model identifier. `GenerativeService` reads
`provider.model` for its stream lifecycle log line (see below); leaving it
`None` just logs `model=None`. Store it as the public `model` attribute (not a
private `_model`) so the service and any audit tooling can read it uniformly.

### EmbeddingProvider

```python
from django_ai_core.generative import EmbeddingProvider

class MyEmbedder(EmbeddingProvider):
    def embedding(self, input, **kwargs): ...        # required
    async def aembedding(self, input, **kwargs): ...  # required
```

Both methods are required.

!!! note "Not wired up yet"

    `EmbeddingProvider` and `resolve_embedding_provider` exist and work, but
    nothing in the library consumes them yet — no shipped provider implements
    embeddings, and the index module still uses the deprecated
    [`LLMService`](legacy-llm-service.md). The interface is ready for a custom
    provider; broader integration is a TODO.

## Translate failures

A provider is the boundary where a backend's raw errors become the library's
[semantic exceptions](index.md#exceptions). Never let a raw SDK / HTTP error
escape — map it to an `AICoreProviderError` subclass and chain the cause:

```python
from django_ai_core.exceptions import (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderResponseError,
)

def completion(self, prompt, **kwargs):
    try:
        resp = self._client.generate(self._model, str(prompt), **kwargs)
    except MyBackendRateLimit as exc:
        raise ProviderRateLimitError(str(exc)) from exc
    except MyBackendTimeout as exc:
        raise ProviderTimeoutError(str(exc)) from exc
    text = resp.text or ""
    if not text:
        raise ProviderResponseError("provider returned an empty response")
    return text
```

Pick the closest semantic type; fall back to `ProviderResponseError` for a
received-but-unusable response and to the base `AICoreProviderError` for anything
unclassifiable. See [`AnyLLMProvider`](anyllm-provider.md) for
a worked example that centralises this in a small mapping table and a context
manager.

## Streaming, usage & cancellation

If your backend streams, override `stream` / `astream` as generators yielding
text deltas. Two concerns beyond yielding text:

### Fill `usage_capture` (best-effort)

`GenerativeService` passes a `UsageCapture` into your stream and reads it after
the stream ends — on clean finish, error, **and** client disconnect — for its
lifecycle log. It is a mutable sink with `input_tokens` / `output_tokens`
(both `int | None`, default `None`):

```python
def stream(self, prompt, *, usage_capture: UsageCapture | None = None, **kwargs):
    for chunk in self._client.stream(self.model, str(prompt), **kwargs):
        if usage_capture is not None and chunk.usage is not None:
            usage_capture.input_tokens = chunk.usage.input_tokens
            usage_capture.output_tokens = chunk.usage.output_tokens
        if chunk.text:
            yield chunk.text
```

Rules:

- **Best-effort, never fatal.** Reading usage must not break the stream — guard
  missing fields; leave them `None` when the backend doesn't supply them.
- **Write synchronously, no `await`.** Fill it as chunks arrive so a stream
  cancelled part-way still leaves whatever counts were seen. Fields the backend
  only emits on a terminal frame stay `None` when the stream is cancelled before
  that frame — that is expected.
- **`usage_capture` may be `None`.** A direct caller might not pass one. Always
  guard `if usage_capture is not None`.

### Cancellation cleanup

When a client disconnects mid-stream, the consumer closes the generator, raising
`GeneratorExit` (async: `aclose()` → `GeneratorExit`, or task cancel →
`asyncio.CancelledError`) at the paused `yield`. The service detects this and
logs `state="cancelled"` — you don't emit anything. But if your `stream` holds a
resource (open socket, SDK context manager), release it in a `try/finally` so it
closes on cancel:

```python
def stream(self, prompt, *, usage_capture: UsageCapture | None = None, **kwargs):
    with self._client.open_stream(self.model, str(prompt)) as s:  # closes on GeneratorExit
        for chunk in s:
            ...
            yield chunk.text
```

Do **not** catch and swallow `GeneratorExit` — re-raise it (a plain
`try/finally`, or `except GeneratorExit: ...; raise`). Swallowing it raises
`RuntimeError: generator ignored GeneratorExit`. Keep cleanup synchronous; an
`await` after the disconnect point races the teardown and may be dropped.

## Wire it up

Point a role at your class by dotted path — `params` become constructor kwargs:

```python
AI_CORE = {
    "GENERATIVE_MODELS": {
        "chat": {
            "provider": "myapp.providers.MyProvider",
            "params": {"model": "my-model-v2"},
        },
    },
}
```

Resolve it like any other provider:

```python
from django_ai_core.generative import GenerativeService, resolve_generative_provider

service = GenerativeService.for_role("chat")           # generic surface
provider = resolve_generative_provider("chat", expect=MyProvider)  # + domain methods
provider.rerank(docs)
```

## Test it

- Assert each backend failure maps to the correct `AICoreProviderError` subclass,
  with the original exception chained (`raise ... from exc`).
- Cover the empty / unusable response path → `ProviderResponseError`.
- If you implement streaming, test the mid-stream error path too, not just setup.
- If you fill `usage_capture`, test that a clean finish populates it and that a
  stream cancelled before the terminal frame leaves the unavailable fields
  `None`.
