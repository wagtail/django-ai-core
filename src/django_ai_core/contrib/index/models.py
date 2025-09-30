import hashlib
from typing import TYPE_CHECKING

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models

from .base import registry

if TYPE_CHECKING:
    from .base import VectorIndex


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
        else:
            return cache_entry, False

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
        except cls.DoesNotExist:
            return None
        else:
            return cache_entry.embedding_vector


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


class ModelSourceIndex(models.Model):
    """Tracks which model instances are indexed in which vector indexes."""

    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey("content_type", "object_id")

    index_name = models.CharField(max_length=255)

    source_id = models.CharField(max_length=255)

    indexed_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["content_type", "object_id"]),
            models.Index(fields=["index_name"]),
            models.Index(fields=["source_id"]),
        ]
        unique_together = [
            ("content_type", "object_id", "index_name", "source_id"),
        ]

    def __str__(self):
        return f"Model Source Index: {self.object_id} in {self.index_name}"

    @classmethod
    def register(cls, obj, index_name, source_id):
        """Register an object as being indexed in the specified index."""
        content_type = ContentType.objects.get_for_model(obj)

        cls.objects.update_or_create(
            content_type=content_type,
            object_id=obj.pk,
            index_name=index_name,
            source_id=source_id,
        )

    @classmethod
    def unregister(cls, obj, index_name=None, source_id=None):
        """Remove registration for an object from one or all indexes."""
        content_type = ContentType.objects.get_for_model(obj)

        filters = {
            "content_type": content_type,
            "object_id": obj.pk,
        }

        if index_name:
            filters["index_name"] = index_name

        if source_id:
            filters["source_id"] = source_id

        cls.objects.filter(**filters).delete()

    @classmethod
    def get_indexed_objects(cls, index_name, source_id=None):
        """Get all objects indexed in the specified index and optionally source."""
        filters = {"index_name": index_name}

        if source_id:
            filters["source_id"] = source_id

        # Get all registrations for this index
        registrations = cls.objects.filter(**filters)

        # Group by content type for efficient fetching
        by_content_type = {}
        for reg in registrations:
            if reg.content_type_id not in by_content_type:
                by_content_type[reg.content_type_id] = []
            by_content_type[reg.content_type_id].append(reg.object_id)

        # Fetch objects by content type
        results = []
        for content_type_id, object_ids in by_content_type.items():
            content_type = ContentType.objects.get_for_id(content_type_id)
            model_class = content_type.model_class()

            # Fetch objects in bulk
            objects = model_class.objects.filter(pk__in=object_ids)
            results.extend(objects)

        return results

    @classmethod
    def get_indexes_for_object(cls, obj) -> list["VectorIndex"]:
        """Get all indexes that contain the specified object."""
        content_type = ContentType.objects.get_for_model(obj)

        registrations = cls.objects.filter(
            content_type=content_type,
            object_id=obj.pk,
        )

        indexes = []
        for registration in registrations:
            indexes.append(registry.get(registration.index_name))

        return indexes
