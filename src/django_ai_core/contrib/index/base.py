from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)
import logging
from queryish import VirtualModel
from .query import QueryHandler

if TYPE_CHECKING:
    from .storage.base import StorageProvider
    from .query import ResultQuerySet
    from .embedding import EmbeddingTransformer
    from .source import Source


logger = logging.getLogger(__name__)


__all__ = [
    "VectorIndex",
    "IndexRegistry",
]


class VectorIndex:
    sources: ClassVar[list["Source"]]
    embedding_transformer: ClassVar["EmbeddingTransformer"]
    storage_provider: ClassVar["StorageProvider"]
    metadata: ClassVar[dict[str, Any] | None] = None
    query_handler: "QueryHandler"

    def __init__(self):
        self.query_handler = QueryHandler()
        self.query_handler.configure(
            storage_provider=self.storage_provider,
            sources=self.sources,
            embedding_transformer=self.embedding_transformer,
        )

    def build(self):
        """
        Build/rebuild the index from configured sources.

        This will:
        1. Get documents from all sources
        2. Process them through the embedding pipeline
        3. Store the embedded documents in the vector storage

        Returns:
            Self for method chaining
        """
        documents = []
        for source in self.sources:
            logger.info(f"Getting documents from source {source.source_id}")
            documents.extend(source.get_documents())

        if not documents:
            logger.warning("No documents provided by sources")
            return self

        # Embed Documents
        logger.info(f"Embedding {len(documents)} documents")
        embedded_documents = self.embedding_transformer.transform(
            documents, batch_size=100
        )

        # Store in vector database
        if embedded_documents:
            logger.info(
                f"Storing {len(embedded_documents)} documents in vector database"
            )
            self.storage_provider.add(embedded_documents)
        else:
            logger.warning("No embedded documents produced by the pipeline")

        return self

    def search(self, query: str) -> "ResultQuerySet":
        """Search the index and return a queryish object of results.
        Args:
            query: The search query string
        Returns:
            ResultQuerySet instance for the search results
        """
        return self.query_handler.search(query)

    def find_similar(self, obj: object) -> "ResultQuerySet":
        """Find objects similar to the given object.
        Args:
            obj: The object to find similar objects to
        Returns:
            ResultQuerySet instance for the search results
        """
        return self.query_handler.find_similar(obj)


class IndexRegistry:
    def __init__(self):
        self._indexes: dict[str, type[VectorIndex]] = {}

    def register(self, slug: str | None = None):
        """Decorator to register an index."""

        def decorator(cls: type[VectorIndex]) -> type[VectorIndex]:
            index_slug = cls.__name__
            self._indexes[index_slug] = cls
            return cls

        return decorator

    def get(self, slug: str) -> type[VectorIndex]:
        if slug not in self._indexes:
            raise KeyError(f"Index '{slug}' not found")
        return self._indexes[slug]

    def list(self) -> dict[str, type[VectorIndex]]:
        return self._indexes.copy()


registry = IndexRegistry()
