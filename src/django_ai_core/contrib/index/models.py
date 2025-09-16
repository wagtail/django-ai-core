import hashlib
from django.db import models


class EmbeddingCache(models.Model):
    """
    Cache for embedded documents to avoid expensive embedding operations.

    Uses content hash for cache key to detect identical content across
    different documents.

    A unique constraint is defined on the content hash and embedding transformer
    ID to ensure that only one cache entry is created per unique content and
    transformer.
    i.e. The same content can be embedded multiple times with different
    transformers, but each embedding will only be cached once.
    """

    # Hash of the chunk content (for cache lookup)
    content_hash = models.CharField(max_length=64, db_index=True)
    # Embeding transformer ID used to generate the embedding
    embedding_transformer_id = models.CharField(max_length=255)

    # Original content that was embedded
    content = models.TextField()

    embedding_vector = models.JSONField()
    embedding_dimensions = models.IntegerField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "django_ai_core_embedding_cache"
        constraints = [
            models.UniqueConstraint(
                fields=["content_hash", "embedding_transformer_id"],
                name="unique_embedding_cache",
            ),
        ]
        indexes = [
            models.Index(fields=["content_hash"]),
            models.Index(fields=["embedding_transformer_id"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"EmbeddingCache({self.content_hash[:12]}...)"

    @classmethod
    def get_content_hash(cls, content: str) -> str:
        """Generate a hash for the given content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def get_or_create_embedding(
        cls,
        *,
        content: str,
        embedding_transformer_id: str,
        embedding_vector: list[float],
    ) -> tuple["EmbeddingCache", bool]:
        """
        Get existing embedding from cache or create new one.

        Returns (embedding_cache, created) tuple.
        """
        content_hash = cls.get_content_hash(content)

        try:
            cache_entry = cls.objects.get(
                content_hash=content_hash,
                embedding_transformer_id=embedding_transformer_id,
            )
            return cache_entry, False
        except cls.DoesNotExist:
            # Create new cache entry
            cache_entry = cls.objects.create(
                content_hash=content_hash,
                content=content,
                embedding_transformer_id=embedding_transformer_id,
                embedding_vector=embedding_vector,
                embedding_dimensions=len(embedding_vector),
            )
            return cache_entry, True

    @classmethod
    def get_cached_embedding(
        cls, *, content: str, embedding_transformer_id: str
    ) -> list[float] | None:
        """
        Get cached embedding for content and model, or None if not found.
        """
        content_hash = cls.get_content_hash(content)

        try:
            cache_entry = cls.objects.get(
                content_hash=content_hash,
                embedding_transformer_id=embedding_transformer_id,
            )
            return cache_entry.embedding_vector
        except cls.DoesNotExist:
            return None


class DocumentEmbedding(models.Model):
    """
    Maps specific document keys to their cached embeddings.

    This allows tracking which documents are using which cached embeddings
    and enables cleanup when documents are deleted.
    """

    # Unique key for the document
    document_key = models.CharField(max_length=500, db_index=True)

    # Reference to the cached embedding
    embedding_cache = models.ForeignKey(
        EmbeddingCache, on_delete=models.CASCADE, related_name="document_mappings"
    )

    # Metadata about when this mapping was created
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "django_ai_core_chunk_embedding"
        unique_together = [["document_key", "embedding_cache"]]
        indexes = [
            models.Index(fields=["document_key"]),
            models.Index(fields=["created_at"]),
        ]

    def __str__(self):
        return f"DocumentEmbedding({self.document_key})"
