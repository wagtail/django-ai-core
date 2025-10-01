# Index Module Overview

The Index Module provides a vector indexing system for Django applications. It enables you to convert Django models into searchable vector representations, supporting semantic search and similarity matching.

## Quick Start

1. Add `django_ai_core.contrib.index` to `INSTALLED_APPS`:

    ```python
    INSTALLED_APPS = [
        # ...
        'django_ai_core.contrib.index',
    ]
    ```

2. Create an `indexes.py` file in your app.
3. Define a new `VectorIndex` subclass in `indexes.py`:

    ```python

    import requests
    from django_ai_core.llm import LLMService
    from django_ai_core.contrib.index import VectorIndex, registry
    from django_ai_core.contrib.index import (
        CachedEmbeddingTransformer,
        CoreEmbeddingTransformer,
    )
    from django_ai_core.contrib.index.source import ModelSource
    from django_ai_core.contrib.index.storage.pgvector import PgVectorProvider

    from .models import MyModel

    @registry.register()
    class MyModelIndex(VectorIndex):
        sources = [
            ModelSource(
                model=MyModel,
            ),
        ]
        storage_provider = PgVectorProvider(model=MediaVectorModel) # TODO: Use something easier to get started with?
        embedding_transformer = CachedEmbeddingTransformer(
            base_transformer=CoreEmbeddingTransformer(
                llm_service=LLMService.create(
                    provider="openai",
                    model="text-embedding-3-small",
                )
            ),
        )
    ```

4. Import `indexes` in your `apps.py` `ready()` function:

    ```python

    from django.apps import AppConfig

    class MyExampleAppConfig(AppConfig):
        label = "myexample"
        name = "myexample"
        verbose_name = "MyExample"

        def ready(self):
            from . import indexes
    ```

5. Build your indexes with `manage.py rebuild_indexes`
6. Query your index with `MyIndex().search_sources("query")`

## Querying Indexes

When indexes are built, source objects are often chunked in to many separate Documents before they are embedded and inserted in to the index.

This means that a query to the underlying index can return multiple Documents from the same source object. For example; if you have a `Book` Django model with a big summary to be embedded, searching the index might return many Documents from the same Book.

This can be fine in some cases, in RAG applications the most relevant chunks are usually what you want, even if they all come from the same source.

In other cases, such as finding similar content, this behaviour can be a hindrance.

To solve this, Vector Indexes provide two query methods depending on your needs:

### Document Search

```python

MyVectorIndex().search_documents("Similar to this")
```

`search_documents` returns a queryset-like interface over Document objects. If the underlying vector provider returns multiple Documents from the same source object, these will all be returned.

This is useful for RAG-like applications where the most relevant chunks are important.

### Source Search

```python

MyVectorIndex().search_sources("Similar to this")
```

When using the `search_sources` method, a Vector Index will attempt to map results from the index back to original source objects, i.e. in the `Book` example, when using `ModelSource(model=Book)`, this method will return a queryset-like interface over `Book` models.

As the underlying storage provider is likely to return multiple Documents for the same source object, this method overfetches Documents to attempt to ensure enough source objects are returned for your query.

This overfetching behaviour can be customised:

```python

MyVectorIndex().search_sources(
    "Similar to this",
    overfetch_multiplier=4,
    max_overfetch_iterations=3
)
```

Where:

-   `overfetch_multiplier` defines how many multiples of the requested limit will be retrieved from the source, e.g. if you request 5 results and provide an `overfetch_multiplier` of 4, 20 Documents will be retrieved from the index internally. The top 5 unique sources from these will then be returned.
-   `max_overfetch_iterations` defines the maximum number of times the underlying search will be repeated to get all-unique source objects, e.g. if the initial search doesn't return enough unique objects, it will be repeated with an increasing number of items up to `max_overfetch_iterations` times.

### Converting Between Result Types

You can convert between result types on an existing queryset:

```python
# Start with document search, convert to sources
docs = MyVectorIndex().search_documents("query")
sources = docs.as_sources()

# Start with source search, convert to documents
sources = MyVectorIndex().search_sources("query")
docs = sources.as_documents()
```

This can be useful in RAG applications where you want to use `Documents` for building context, but then present source objects to users as the 'Sources referenced'.
