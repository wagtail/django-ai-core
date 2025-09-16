from typing import Type, Generator, TYPE_CHECKING

from pgvector.django import VectorField, CosineDistance

from .base import StorageProvider, StorageQuerySet, StorageDocument
from ..schema import EmbeddedDocument

if TYPE_CHECKING:
    from .pgvector_model import PgVectorModelMixin


class PgVectorQuerySet(StorageQuerySet["PgVectorProvider"]):
    """QuerySet implementation for PgVectorProvider."""

    def get_instance(self, model_instance: "PgVectorModelMixin") -> StorageDocument:
        """Convert a Django model instance to a StorageDocument."""
        if self.model:
            return self.model(
                document_key=model_instance.document_key,
                content=model_instance.content,
                metadata=model_instance.metadata,
            )
        else:
            # Create a pseudo-StorageDocument from the model instance
            class PseudoStorageDocument:
                document_key = model_instance.document_key
                content = model_instance.content
                metadata = model_instance.metadata
                vector = model_instance.vector

            return PseudoStorageDocument()

    def run_query(self) -> Generator[StorageDocument, None, None]:
        """Execute the query and return the results."""
        if not self.storage_provider:
            raise ValueError("Storage provider is required")

        filter_map = {filter[0]: filter[1] for filter in self.filters}

        embedding = filter_map.pop("embedding", None)
        if embedding is None:
            raise ValueError("embedding filter is required")

        if self.ordering:
            raise NotImplementedError("Ordering is not supported for querying")

        model = self.storage_provider.model
        if not model:
            raise ValueError("Model class is required")

        # Get the field name for the embedding vector
        embedding_field = model.embedding_field

        # Build the query with similarity search
        # Default to cosine similarity, but can be configurable in the future
        queryset = model.objects.order_by(CosineDistance(embedding_field, embedding))

        # Apply metadata filters if any
        for key, value in filter_map.items():
            queryset = queryset.filter(**{f"metadata__{key}": value})

        # Apply top_k limit
        queryset = queryset[: self._top_k]

        for instance in queryset:
            yield self.get_instance(instance)


class PgVectorProvider(StorageProvider):
    """
    Vector storage using PostgreSQL with pgvector extension.

    This storage provider allows storing vector embeddings in a PostgreSQL database
    using a custom Django model that includes the PgVectorModelMixin.

    Example:
        class MyVectorModel(PgVectorModelMixin, models.Model):
            embedding = VectorField(dimensions=768)

            class Meta:
                app_label = 'myapp'

        # Create the storage provider with the model
        storage = PgVectorProvider(model=MyVectorModel)
    """

    queryset_cls = PgVectorQuerySet

    def __init__(
        self,
        model: Type["PgVectorModelMixin"],
    ):
        """
        Initialize the PgVectorProvider.

        Args:
            model: A Django model class that includes the PgVectorModelMixin.
        """
        self.model = model
        required_fields = [
            "document_key",
            "content",
            "metadata",
            "vector",
        ]

        # Verify that the model has the required fields
        for field in required_fields:
            if not hasattr(self.model, field):
                raise ValueError(
                    f"Model class {self.model.__name__} must include '{field}' field"
                )

        # Verify that the embedding field is a VectorField
        field = self.model._meta.get_field(self.model.embedding_field)
        if not isinstance(field, VectorField):
            raise ValueError(
                f"Field '{self.model.embedding_field}' must be a VectorField, got {type(field)}"
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
                instance = self.model.objects.get(document_key=document.document_key)
                # Update the existing instance
                instance.content = document.content
                instance.metadata = document.metadata
                instance.vector = document.vector
            except self.model.DoesNotExist:
                # Create a new instance
                instance = self.model(
                    document_key=document.document_key,
                    content=document.content,
                    metadata=document.metadata,
                )
                instance.vector = document.vector

            instances.append(instance)

        # Save all instances using bulk_create and bulk_update
        existing_keys = [i.document_key for i in instances if i.pk]
        new_instances = [i for i in instances if i.pk not in existing_keys]

        # Bulk update existing instances
        if existing_keys:
            existing_instances = [i for i in instances if i.pk in existing_keys]
            self.model.objects.bulk_update(
                existing_instances,
                ["content", "metadata", self.model.embedding_field],
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
