from django_ai_core.llm import LLMService
from django_ai_core.contrib.index import VectorIndex, registry
from django_ai_core.contrib.index import (
    CachedEmbeddingTransformer,
    CoreEmbeddingTransformer,
)
from django_ai_core.contrib.index.source import ModelSource
from django_ai_core.contrib.index.storage import PgVectorProvider

from .models import Book, Film, VideoGame, MediaVectorModel


@registry.register()
class MediaIndex(VectorIndex):
    sources = [
        ModelSource(
            model=Book,
            content_fields=["title", "description"],
            metadata_fields=["description"],
        ),
        ModelSource(model=Film),
        ModelSource(model=VideoGame),
    ]
    storage_provider = PgVectorProvider(model=MediaVectorModel)
    embedding_transformer = CachedEmbeddingTransformer(
        base_transformer=CoreEmbeddingTransformer(
            llm_service=LLMService(
                provider="openai",
                model="text-embedding-3-small",
            )
        ),
    )
