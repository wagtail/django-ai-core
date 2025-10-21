from typing import TYPE_CHECKING, Generator, Type, cast

from ...schema import EmbeddedDocument
from ..base import BaseStorageDocument, BaseStorageQuerySet, StorageProvider

if TYPE_CHECKING:
    from .models import BasePgVectorEmbedding, PgvectorEmbeddingQuerySet


class PgVectorQuerySet(BaseStorageQuerySet["PgVectorProvider"]):
    """QuerySet implementation for PgVectorProvider."""

    def get_instance(self, val: "BasePgVectorEmbedding") -> BaseStorageDocument:
        """Convert a Django model instance to a BaseStorageDocument."""
        return self.model(
            document_key=val.document_key,
            content=val.content,
            metadata=val.metadata,
            score=1 - val.distance,
        )

    def run_query(self) -> Generator[BaseStorageDocument, None, None]:
        """Execute the query and return the results."""
        if not self.storage_provider:
            raise ValueError("Storage provider is required")

        storage_provider = self.storage_provider

        filter_map = {filter[0]: filter[1] for filter in self.filters}

        embedding = filter_map.pop("embedding", None)
        if embedding is None:
            raise ValueError("embedding filter is required")

        if self.ordering:
            raise NotImplementedError("Ordering is not supported for querying")

        model = storage_provider.model
        if not model:
            raise ValueError("Model class is required")

        queryset = model.objects.filter(index_name=storage_provider.index_name)
        queryset = cast("PgvectorEmbeddingQuerySet", queryset)

        queryset = queryset.annotate_with_distance(embedding).order_by("distance")

        # Apply metadata filters if any
        for key, value in filter_map.items():
            queryset = queryset.filter(**{f"metadata__{key}": value})

        queryset = queryset[self.start : self.stop]

        for instance in queryset:
            yield self.get_instance(instance)


class PgVectorProvider(StorageProvider):
    """
    Vector storage using PostgreSQL with pgvector extension.
    """

    base_queryset_cls = PgVectorQuerySet

    def __init__(self, *, model: Type["BasePgVectorEmbedding"] | None = None, **kwargs):
        """
        Initialize the PgVectorProvider.

        Args:
            model: A Django model class that subclasses BasePgVectorEmbedding.
        """
        super().__init__(**kwargs)
        if model:
            self.model = model
        else:
            from .models import PgVectorEmbedding

            self.model = PgVectorEmbedding

        required_fields = [
            "index_name",
            "document_key",
            "content",
            "metadata",
            "vector",
        ]

        for field in required_fields:
            if not hasattr(self.model, field):
                raise ValueError(
                    f"Model class {self.model.__name__} must include '{field}' field"
                )

    def add(self, documents: list[EmbeddedDocument]) -> None:
        """
        Store documents in the PostgreSQL database.

        Args:
            documents: List of embedded documents to store.
        """
        # Create and save model instances for each document
        instances = []
        for document in documents:
            # Check if the document already exists
            try:
                instance = self.model.objects.get(
                    index_name=self.index_name, document_key=document.document_key
                )
                # Update the existing instance
                instance.content = document.content
                instance.metadata = document.metadata
                instance.vector = document.vector
            except self.model.DoesNotExist:
                # Create a new instance
                instance = self.model(
                    index_name=self.index_name,
                    document_key=document.document_key,
                    content=document.content,
                    metadata=document.metadata,
                )
                instance.vector = document.vector

            instances.append(instance)

        # Save all instances using bulk_create and bulk_update
        existing_keys = [i.document_key for i in instances if not i._state.adding]
        new_instances = [i for i in instances if i.pk not in existing_keys]

        # Bulk update existing instances
        if existing_keys:
            existing_instances = [i for i in instances if i.pk in existing_keys]
            self.model.objects.bulk_update(
                existing_instances,
                ["content", "metadata", "vector"],
            )

        # Bulk create new instances
        if new_instances:
            self.model.objects.bulk_create(new_instances)

    def delete(self, document_keys: list[str]) -> None:
        """
        Delete documents by their keys.

        Args:
            document_keys: List of document keys to delete.
        """
        self.model.objects.filter(document_key__in=document_keys).delete()

    def clear(self) -> None:
        """Clear all documents from the database."""
        self.model.objects.all().delete()
