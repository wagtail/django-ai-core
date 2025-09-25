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
    from django_ai_core.contrib.index.storage import PgVectorProvider

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
6. Query your index with `MyIndex().search("query")`
