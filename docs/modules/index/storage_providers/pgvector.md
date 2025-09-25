# pgvector

The pgvector storage provider uses the [pgvector](https://github.com/pgvector/pgvector) PostgreSQL extension to store your vector embeddings and enable querying across them.

## Requirements

To use the `pgvector` storage provider you must:

-   Be using a PostgreSQL database for your Django application
-   Have the `pgvector` extension available on your PostgreSQL database
-   Have the `pgvector` Python package installed

## Usage

The pgvector provider requires a Django model for a table where it can store embeddings. This package provides one you can use by adding `django_ai_core.contrib.index.storage.pgvector` to your `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    'django_ai_core',
    'django_ai_core.contrib.index',
    'django_ai_core.contrib.index.storage.pgvector'
]
```

This will add a new model to your project which is used by the pgvector Provider by default.

!!! note "Custom models"

    See [Custom embedding models](#custom-embedding-models) for information on how and why you can provide your own model instead.

Instantiate a `PgVectorProvider` with:

```python

from django_ai_core.contrib.index.storage.pgvector import PgVectorProvider

provider = PgVectorProvider()
```

## Custom embedding models

While the `PgVectorProvider` provides a model by default, this model uses a variable length vector field - this means that it can store vectors of varying lengths, but means some features like indexes are not available.

To optimise performance you may want to create your own embedding model:

```python
from django_ai_core.contrib.index.storage.pgvector.models import BasePgVectorEmbedding
from pgvector.django import VectorField
from pgvector.django import HnswIndex

class CustomPgVectorEmbedding(BasePgVectorEmbedding):
    vector = VectorField(dimensions=1024)

     class Meta:
        indexes = [
            HnswIndex(
                name='vector_index',
                fields=['vectoro'],
                m=16,
                ef_construction=64,
                opclasses=['vector_l2_ops']
            )
        ]
```

This can then be passed to the provider when instantiating:

```python

from django_ai_core.contrib.index.storage.pgvector import PgVectorProvider
from .models import CustomPgVectorEmbedding

provider = PgVectorProvider(model=CustomPgVectorEmbedding)
```

In your migration for your custom model, add the `VectorExtension` migration operation to ensure the `pgvector` extension is enabled on your database:

```
from pgvector.django import VectorExtension
from django.db import migrations, models


class Migration(migrations.Migration):
    operations = [
        VectorExtension(),
        # ...your operations
    ]
```
