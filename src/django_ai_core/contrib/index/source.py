from typing import TYPE_CHECKING, Iterable, Protocol, TypeVar, runtime_checkable

from django.db import models
from django.db.models import QuerySet

from .chunking import ChunkTransformer, SimpleChunkTransformer
from .schema import Document

if TYPE_CHECKING:
    from .base import VectorIndex


ObjectType = TypeVar("ObjectType")

DEFAULT_DOCUMENT_SIZE = 2000


@runtime_checkable
class HasPostIndexUpdateHook(Protocol):
    def post_index_update(self, index: "VectorIndex"):
        """Called after an index using this source is built or updated"""
        ...


@runtime_checkable
class Source(Protocol):
    """Base source for providing documents for the index."""

    @property
    def source_id(self) -> str:
        """Get unique identifier for this source."""
        return self.__class__.__name__

    def get_documents(self) -> Iterable[Document]:
        """Get all documents from this source."""
        ...

    def provides_document(self, document: Document) -> bool:
        """Check if the given Document is provided by this source"""
        ...


@runtime_checkable
class ObjectSource(Source, Protocol):
    """A Source that adapts a different object, turning it in to a Document, and supporting turning
    Documents back to the original object."""

    def objects_to_documents(
        self, obj: object | Iterable[object]
    ) -> Iterable[Document]:
        """Get documents for the given objects."""
        ...

    def objects_from_documents(self, documents: Iterable[Document]) -> Iterable[object]:
        """Convert documents back to original objects."""
        ...

    def provides_object(self, obj: object) -> bool:
        """Check if the given object belongs to this source."""
        ...


class ModelSource(ObjectSource):
    """Source for Django models with queryset support."""

    def __init__(
        self,
        queryset: QuerySet | None = None,
        model: type[models.Model] | None = None,
        content_fields: list[str] | None = None,
        metadata_fields: list[str] | None = None,
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

        self.content_fields = content_fields
        self.metadata_fields = metadata_fields

    def _get_content_fields(self, obj: models.Model) -> list[str]:
        """Get the list of fields to use for content extraction."""
        if self.content_fields is not None:
            return self.content_fields

        # If no fields specified, use all concrete fields on the model
        # that have reasonable string representations
        field_names = []
        for field in obj._meta.get_fields():
            # Skip many-to-many and reverse relations
            if field.is_relation and (field.many_to_many or field.one_to_many):
                continue

            # Include concrete fields and forward relations (like ForeignKey)
            if field.concrete or (
                (field.is_relation and field.one_to_one) or field.many_to_one
            ):
                field_names.append(field.name)

        return field_names

    def _get_field_value(self, obj: models.Model, field_name: str) -> str | None:
        # Get the field value directly
        field_value = getattr(obj, field_name)
        # Handle callable fields (methods, properties)
        if callable(field_value):
            field_value = field_value()

        # Skip None values
        if field_value is None:
            return None

        # Format the value
        if isinstance(field_value, str):
            final_value = field_value
        else:
            final_value = "\n".join(str(v) for v in [field_value] if v is not None)

        return final_value

    def get_content(self, obj: models.Model) -> str:
        """Extract text content from model instance."""
        content = []
        fields = self._get_content_fields(obj)

        for field_name in fields:
            try:
                field_value = self._get_field_value(obj, field_name)

                if field_value is None:
                    continue

                content.append(field_value)

            except (AttributeError, TypeError):
                # Skip fields that don't exist or cause errors
                continue

        return "\n".join(content)

    def get_metadata(self, obj: models.Model) -> dict:
        """Extract metadata from model instance. Override this method to add additional metadata."""
        metadata_fields = (
            {
                field_name: self._get_field_value(obj, field_name)
                for field_name in self.metadata_fields
            }
            if self.metadata_fields
            else {}
        )
        return {
            "model": obj._meta.label,
            "pk": obj.pk,
            "source_id": self.source_id,
            **metadata_fields,
        }

    @property
    def source_id(self) -> str:
        """Use Django model label as source ID."""
        return self.model._meta.label

    def provides_object(self, obj: object) -> bool:
        """Check if the given object belongs to this source."""
        return self.model is type(obj)

    def provides_document(self, document: Document) -> bool:
        return document.document_key.split(":")[0] == self.source_id

    def get_document_key(self, obj, chunk) -> str:
        return f"{self.source_id}:{obj.pk}:{chunk}"

    def _object_to_documents(self, obj: models.Model) -> Iterable[Document]:
        if not self.provides_object(obj):
            raise ValueError("Object does not belong to this source")

        metadata = self.get_metadata(obj)
        content = self.get_content(obj)

        for chunk, document in enumerate(self.chunk_transformer.transform(content)):
            yield Document(
                document_key=self.get_document_key(obj, chunk),
                content=document,
                metadata=metadata,
            )

    def get_documents(self) -> Iterable[Document]:
        """Convert querysets to documents."""
        for obj in self.queryset:
            yield from self.objects_to_documents(obj)

    def objects_to_documents(
        self, objs: models.Model | Iterable[models.Model]
    ) -> Iterable[Document]:
        """Get documents for the given objects."""

        if isinstance(objs, models.Model):
            objs = [objs]

        for obj in objs:
            yield from self._object_to_documents(obj)

    def objects_from_documents(
        self, documents: Iterable[Document]
    ) -> Iterable[models.Model]:
        """Convert documents back to Django model instances if they were created by this source.
        Returns objects in same order as provided Documents."""

        pks = []
        for document in documents:
            if self.provides_document(document):
                pks.append(document.metadata["pk"])

        if not pks:
            return []

        # Deduplicate pks for more efficient lookup
        pks = list(set(pks))

        objects = list(self.model.objects.filter(pk__in=pks))
        pk_to_object = {obj.pk: obj for obj in objects}

        for document in documents:
            if self.provides_document(document):
                pk = document.metadata["pk"]
                if pk in pk_to_object:
                    yield pk_to_object[pk]

    def post_index_update(self, index):
        from .models import ModelSourceIndex

        ModelSourceIndex.objects.filter(
            index_name=index, source_id=self.source_id
        ).delete()

        for obj in self.queryset:
            ModelSourceIndex.register(obj, index, self.source_id)
