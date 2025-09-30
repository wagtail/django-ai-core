"""
Embedding cache abstraction for vector indexing.

Provides a caching layer that sits between document transformers and embedding
generation to avoid expensive embedding operations for duplicate content.
"""

import logging
from abc import ABC, abstractmethod

from django_ai_core.contrib.index.schema import Document, EmbeddedDocument

from .embedding import EmbeddingTransformer

logger = logging.getLogger(__name__)


class EmbeddingCacheBackend(ABC):
    """Abstract base class for embedding cache backends."""

    @abstractmethod
    def get_embedding(self, content: str, transformer_id: str) -> list[float] | None:
        """Get cached embedding for content and model, or None if not found."""
        pass

    @abstractmethod
    def store_embedding(
        self, content: str, transformer_id: str, embedding: list[float]
    ) -> None:
        """Store embedding in cache."""
        pass

    def get_embeddings_batch(
        self, contents: list[str], transformer_id: str
    ) -> dict[str, list[float]]:
        """Get cached embeddings for multiple contents. Returns dict mapping content to embedding."""
        result = {}
        for content in contents:
            embedding = self.get_embedding(content, transformer_id)
            if embedding is not None:
                result[content] = embedding
        return result

    def store_embeddings_batch(
        self, content_embeddings: dict[str, list[float]], transformer_id: str
    ) -> None:
        """Store multiple embeddings in cache."""
        for content, embedding in content_embeddings.items():
            self.store_embedding(content, transformer_id, embedding)

    @abstractmethod
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        pass


class DjangoEmbeddingCacheBackend(EmbeddingCacheBackend):
    """Django model-based embedding cache backend."""

    def _get_cache_model(self):
        """Get the cache model."""
        try:
            from .models import EmbeddingCache
        except ImportError as e:
            raise ImportError(
                "Django is not properly configured. Make sure Django settings are loaded "
                "and 'django_ai_core.contrib.index' is in INSTALLED_APPS."
            ) from e
        else:
            return EmbeddingCache

    def get_embedding(self, content: str, transformer_id: str) -> list[float] | None:
        """Get cached embedding for content and model."""
        return self._get_cache_model().get_cached_embedding(
            content=content, embedding_transformer_id=transformer_id
        )

    def store_embedding(
        self, content: str, transformer_id: str, embedding: list[float]
    ) -> None:
        """Store embedding in cache."""
        self._get_cache_model().get_or_create_embedding(
            content=content,
            embedding_transformer_id=transformer_id,
            embedding_vector=embedding,
        )

    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._get_cache_model().objects.all().delete()


class CachedEmbeddingTransformer(EmbeddingTransformer):
    """
    Embedding transformer with caching functionality that inherits from EmbeddingTransformer.

    This provides a consistent interface that can be composed with other transformers.
    """

    def __init__(
        self,
        base_transformer: "EmbeddingTransformer",
        cache_backend: EmbeddingCacheBackend | None = None,
    ):
        """
        Initialize cached embedding transformer.

        Args:
            base_transformer: The actual embedding transformer to wrap
            cache_backend: Cache backend to use (defaults to Django backend)
        """
        self.base_transformer = base_transformer
        self.cache_backend = cache_backend or DjangoEmbeddingCacheBackend()
        self.cache_hits = 0

    @property
    def transformer_id(self) -> str:
        """Get unique identifier for this transformer."""
        return f"cached_{self.base_transformer.transformer_id}"

    def embed_string(self, text: str) -> list[float] | None:
        """Embed a string using the cache backend."""
        cached = self.cache_backend.get_embedding(text, self.transformer_id)
        if cached is not None:
            return cached

        result = self.base_transformer.embed_string(text)
        if result:
            self.cache_backend.store_embedding(text, self.transformer_id, result)
            return result

    def embed_documents(
        self, documents: list["Document"], *, batch_size: int = 100
    ) -> list["EmbeddedDocument"]:
        """Transform multiple documents with caching.

        Args:
            documents: List of documents to transform
            batch_size: Number of documents to embed in each batch when processing uncached documents

        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return []

        # Get all content strings for batch cache lookup
        contents = [document.content for document in documents]
        cached_embeddings = self.cache_backend.get_embeddings_batch(
            contents, self.base_transformer.transformer_id
        )

        # Separate cached and uncached documents
        cached_documents: list[tuple[int, EmbeddedDocument]] = []
        uncached_documents: list[Document] = []
        uncached_indices = []

        for i, document in enumerate(documents):
            if document.content in cached_embeddings:
                self.cache_hits += 1
                logger.debug(f"Cache hit for document {document.document_key}")
                cached_documents.append(
                    (
                        i,
                        EmbeddedDocument(
                            document_key=document.document_key,
                            content=document.content,
                            metadata=document.metadata,
                            vector=cached_embeddings[document.content],
                        ),
                    )
                )
            else:
                logger.debug(f"Cache miss for document {document.document_key}")
                uncached_documents.append(document)
                uncached_indices.append(i)

        # Process uncached documents in batch
        if uncached_documents:
            embedded_documents = self.base_transformer.embed_documents(
                uncached_documents, batch_size=batch_size
            )

            # Store new embeddings in cache
            new_embeddings = {}
            for document in embedded_documents:
                if document.vector is not None:
                    new_embeddings[document.content] = document.vector

            if new_embeddings:
                self.cache_backend.store_embeddings_batch(
                    new_embeddings, self.base_transformer.transformer_id
                )
        else:
            embedded_documents = []

        # Combine cached and newly embedded documents in original order
        result: list[EmbeddedDocument | None] = [None] * len(documents)

        # Place cached documents
        for i, document in cached_documents:
            result[i] = document

        # Place newly embedded documents
        for i, document in zip(uncached_indices, embedded_documents, strict=False):
            result[i] = document

        return [document for document in result if document is not None]
