from abc import ABC, abstractmethod
from typing import TypeVar
from django.db import models
from django.db.models import QuerySet

from .schema import Document
from .chunking import ChunkTransformer, SimpleChunkTransformer


ObjectType = TypeVar("ObjectType")

DEFAULT_DOCUMENT_SIZE = 2000


class Source(ABC):
    """Base source for providing documents for the index."""

    @property
    def source_id(self) -> str:
        """Get unique identifier for this source."""
        return self.__class__.__name__

    @abstractmethod
    def object_belongs_to_source(self, obj: object) -> bool:
        """Check if the given object belongs to this source."""
        pass

    @abstractmethod
    def get_documents_for_object(self, obj: object) -> list[Document]:
        """Get documents for the given object."""
        pass

    @abstractmethod
    def get_documents(self) -> list[Document]:
        """Get all documents from this source."""
        pass

    @abstractmethod
    def get_objects_from_documents(self, documents: list[Document]) -> list[object]:
        """Convert documents back to original objects."""
        pass


class ModelSource(Source):
    """Source for Django models with queryset support."""

    def __init__(
        self,
        queryset: QuerySet | None = None,
        model: type[models.Model] | None = None,
        fields: list[str] | None = None,
        chunk_transformer: ChunkTransformer | None = None,
    ):
        if queryset is not None:
            self.queryset = queryset
            self.model = queryset.model
        elif model is not None:
            self.model = model
            self.queryset = model.objects.all()
        else:
            raise ValueError("Either queryset or model must be provided")

        if chunk_transformer is None:
            self.chunk_transformer = SimpleChunkTransformer()

        self.fields = fields

    def _get_fields(self, obj: models.Model) -> list[str]:
        """Get the list of fields to use for content extraction."""
        if self.fields is not None:
            return self.fields

        # If no fields specified, use all concrete fields on the model
        # that have reasonable string representations
        field_names = []
        for field in obj._meta.get_fields():
            # Skip many-to-many and reverse relations
            if field.is_relation and (field.many_to_many or field.one_to_many):
                continue

            # Include concrete fields and forward relations (like ForeignKey)
            if field.concrete or (
                field.is_relation and field.one_to_one or field.many_to_one
            ):
                field_names.append(field.name)

        return field_names

    def get_content(self, obj: models.Model) -> str:
        """Extract text content from model instance."""
        content = []
        fields = self._get_fields(obj)

        for field_name in fields:
            try:
                # Get the field value directly
                field_value = getattr(obj, field_name)
                # Handle callable fields (methods, properties)
                if callable(field_value):
                    field_value = field_value()

                # Skip None values
                if field_value is None:
                    continue

                # Format the value
                if isinstance(field_value, str):
                    final_value = field_value
                else:
                    final_value = "\n".join(
                        str(v) for v in [field_value] if v is not None
                    )

                content.append(final_value)
            except (AttributeError, TypeError):
                # Skip fields that don't exist or cause errors
                continue

        return "\n".join(content)

    def get_metadata(self, obj: models.Model) -> dict:
        """Extract metadata from model instance. Override this method to add additional metadata."""
        return {
            "model": obj._meta.label,
            "pk": obj.pk,
            "source_id": self.source_id,
        }

    @property
    def source_id(self) -> str:
        """Use Django model label as source ID."""
        return self.model._meta.label

    def object_belongs_to_source(self, obj: object) -> bool:
        """Check if the given object belongs to this source."""
        return self.model is type(obj)

    def get_documents_for_object(self, obj: models.Model) -> list[Document]:
        """Get documents for the given object."""

        if not self.object_belongs_to_source(obj):
            raise ValueError("Object does not belong to this source")

        metadata = self.get_metadata(obj)
        content = self.get_content(obj)

        documents = []

        for idx, document in enumerate(self.chunk_transformer.transform(content)):
            documents.append(
                Document(
                    document_key=f"{obj._meta.label}:{obj.pk}:{idx}",
                    content=document,
                    metadata=metadata,
                )
            )
        return documents

    def get_documents(self) -> list[Document]:
        """Convert querysets to documents."""
        documents = []

        for obj in self.queryset:
            documents.extend(self.get_documents_for_object(obj))

        return documents

    def get_objects_from_documents(
        self, documents: list[Document]
    ) -> list[models.Model]:
        """Convert documents back to Django model instances if they were created by this source."""

        def _document_is_for_model(document: Document) -> bool:
            return (
                "pk" in document.metadata
                and "source_id" in document.metadata
                and document.metadata["source_id"] == self.source_id
            )

        pks = []
        for document in documents:
            if _document_is_for_model(document):
                pks.append(document.metadata["pk"])

        if not pks:
            return []

        # Deduplicate pks
        pks = list(set(pks))

        # Bulk fetch all objects in a single query
        objects = list(self.model.objects.filter(pk__in=pks))

        # Create a mapping for quick lookup
        pk_to_object = {obj.pk: obj for obj in objects}

        # Return objects in the same order as documents, skipping missing ones
        result = []
        for document in documents:
            if _document_is_for_model(document):
                pk = document.metadata["pk"]
                if pk in pk_to_object:
                    result.append(pk_to_object[pk])

        return result
