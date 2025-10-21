from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Iterable, Iterator, TypeVar

from queryish import Queryish, VirtualModel

from ..schema import EmbeddedDocument

StorageProviderType = TypeVar("StorageProviderType", bound="StorageProvider")


class BaseStorageQuerySet(Queryish, Generic[StorageProviderType]):
    """Base Queyrish QuerySet. Subclasses are generated dynamically by Queryish."""

    # Defaults to None even though this isn't a valid type as Queryish
    # uses 'hasattr' to check if it can copy a Meta attribute from the Virtual Model
    # Subclasses do not need to specify storage_provider or model - these
    # are automatically added to the generated QuerySet class by Queryish,
    # based on the values provided in BaseStorageDocument.Meta
    storage_provider: StorageProviderType = None  # type: ignore
    model: type["BaseStorageDocument"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop = 20

    def run_query(self) -> Iterator["BaseStorageDocument"]:
        """Execute the query and return the results."""
        raise NotImplementedError


class BaseStorageDocument(VirtualModel):
    """Base virtual model for Documents in storage backends. Subclasses are generated dynamically by StorageProviders."""

    base_query_class = BaseStorageQuerySet
    pk_field_name = "document_key"

    document_key: str
    content: str
    metadata: dict[str, Any]
    score: float = 0

    class Meta:
        fields = ["document_key", "content", "metadata", "score"]
        storage_provider: "StorageProvider"

    def __str__(self):
        return self.document_key


class StorageProvider(ABC):
    """Base class for vector storage backends."""

    base_queryset_cls: ClassVar[type[BaseStorageQuerySet]]

    def __init__(self, *, index_name: str | None = None, **kwargs):
        self.index_name = index_name

    @property
    def document_cls(self):
        """Build a document class for this storage provider."""
        meta = type(
            "Meta",
            (BaseStorageDocument.Meta,),
            {
                "storage_provider": self,
            },
        )

        # Determine document class name
        document_class_name = f"{self.__class__.__name__}Document"
        if self.__class__.__name__.endswith("Provider"):
            document_class_name = self.__class__.__name__.replace(
                "Provider", "Document"
            )

        return type(
            document_class_name,
            (BaseStorageDocument,),
            {"Meta": meta, "base_query_class": self.base_queryset_cls},
        )

    @abstractmethod
    def add(self, documents: Iterable["EmbeddedDocument"]):
        """Store documents in the vector database."""
        pass

    @abstractmethod
    def delete(self, document_keys: Iterable[str]):
        """Delete documents by their keys."""
        pass

    @abstractmethod
    def clear(self):
        """Clear the vector database."""
        ...

    @property
    def objects(self):
        return self.document_cls().objects

    @property
    def Document(self):
        return self.document_cls
