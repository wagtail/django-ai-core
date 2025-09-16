from typing import Protocol, ClassVar, Generic, TypeVar, Generator, Any

from ..schema import EmbeddedDocument
from queryish import VirtualModel, Queryish

StorageProviderType = TypeVar("StorageProviderType", bound="StorageProvider")


class StorageQuerySet(Queryish, Generic[StorageProviderType]):
    """Queryish interface for storage backends."""

    storage_provider: StorageProviderType | None = None
    model: type["StorageDocument"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._top_k: int = 5

    def top_k(self, k: int) -> "StorageQuerySet":
        """Limit the number of results to the top k."""
        clone = self.clone()
        clone._top_k = k
        return clone

    def run_query(self) -> Generator["StorageDocument", None, None]:
        """Execute the query and return the results."""
        raise NotImplementedError


class StorageDocument(VirtualModel):
    """Virtual model for Documents in storage backends."""

    base_query_class = StorageQuerySet

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

    queryset_cls: ClassVar[type[StorageQuerySet]]

    def add(self, documents: list["EmbeddedDocument"]):
        """Store documents in the vector database."""
        ...

    def delete(self, document_keys: list[str]):
        """Delete documents by their keys."""
        ...

    def clear(self):
        """Clear the vector database."""
        ...

    def document_cls(self) -> type[StorageDocument]:
        """Build a document class for this storage provider."""
        meta = type(
            "Meta",
            (StorageDocument.Meta,),
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

        document_cls = type(
            document_class_name,
            (StorageDocument,),
            {"Meta": meta, "base_query_class": self.queryset_cls},
        )
        return document_cls

    def search(self, query_embedding: list[float]) -> StorageQuerySet:
        """Query the vector database for similar documents."""
        return self.document_cls().objects.filter(embedding=query_embedding)
