from typing import Protocol, ClassVar, Generic, TypeVar, Generator, Any

from ..schema import EmbeddedDocument
from queryish import VirtualModel, Queryish

StorageProviderType = TypeVar("StorageProviderType", bound="StorageProvider")

"""
Storage providers implement an interface for storing documents in vector storage backends, as well as for querying from them.

The query interface uses Queryish so that results behave like Django QuerySets.

There's some auto-class generation oddities going on here so for clarity:

- Each storage provider implements
    - a subclass of StorageProvider (defining the provider configuration and how to add/delete data)
    - a subclass of BaseStorageQuerySet (defining how to query data)
- The StorageProvider class is provided with a `base_queryset_cls` classvar which tells it what base queryset to use.
- The StorageProvider's `document_cls` property generates a Queryish Virtual Model from BaseStorageDocument, setting a `base_queryset_class`
- When a VirtualModel (BaseStorageDocument in this case) class is created, Queryish generates a new queryset class based on `base_queryset_class`.

"""


class BaseStorageQuerySet(Queryish, Generic[StorageProviderType]):
    """Base Queyrish QuerySet. Subclasses are generated dynamically by Queryish."""

    storage_provider: StorageProviderType = None  # Defaults to None even though this isn't a valid type as Queryish uses 'hasattr' to check if it can copy a Meta attribute from the Virtual Model
    model: type["BaseStorageDocument"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._top_k: int = 5

    def top_k(self, k: int) -> "BaseStorageQuerySet":
        """Limit the number of results to the top k."""
        clone = self.clone()
        clone._top_k = k
        return clone

    def run_query(self) -> Generator["BaseStorageDocument", None, None]:
        """Execute the query and return the results."""
        raise NotImplementedError


class BaseStorageDocument(VirtualModel):
    """Base virtual model for Documents in storage backends. Subclasses are generated dynamically by StorageProviders."""

    base_query_class = BaseStorageQuerySet
    pk_field_name = "document_key"

    document_key: str
    content: str
    metadata: dict[str, Any]

    class Meta:
        fields = ["document_key", "content", "metadata"]
        storage_provider: "StorageProvider"

    def __str__(self):
        return self.document_key


class StorageProvider(Protocol):
    """Base class for vector storage backends."""

    base_queryset_cls: ClassVar[type[BaseStorageQuerySet]]

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

    def add(self, documents: list["EmbeddedDocument"]):
        """Store documents in the vector database."""
        ...

    def delete(self, document_keys: list[str]):
        """Delete documents by their keys."""
        ...

    def clear(self):
        """Clear the vector database."""
        ...

    @property
    def objects(self):
        return self.document_cls().objects

    @property
    def Document(self):
        return self.document_cls
