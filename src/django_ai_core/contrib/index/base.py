import logging
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

from django.utils.text import slugify

from .source import HasPostIndexUpdateHook

if TYPE_CHECKING:
    from .embedding import EmbeddingTransformer
    from .query import QueryHandler
    from .schema import Document
    from .source import Source
    from .storage.base import BaseStorageQuerySet, StorageProvider


logger = logging.getLogger(__name__)


class VectorIndex:
    sources: ClassVar[list["Source"]]
    embedding_transformer: ClassVar["EmbeddingTransformer"]
    storage_provider: ClassVar["StorageProvider"]
    query_handler: "QueryHandler"
    metadata: ClassVar[dict[str, Any] | None] = None

    @property
    def index_id(self):
        class_name_slug = slugify(self.__class__.__name__)
        return class_name_slug

    def __init__(self):
        from .query import QueryHandler

        self.query_handler = QueryHandler()
        self.query_handler.configure(
            storage_provider=self.storage_provider,
            sources=self.sources,
            embedding_transformer=self.embedding_transformer,
        )

        # Set the storage provider index name from the index ID
        self.storage_provider.index_name = f"{self.index_id}_index"

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

        self.update(documents)

        return self

    def update(self, documents: Iterable["Document"]):
        # Embed documents
        logger.info("Embedding documents")
        embedded_documents = self.embedding_transformer.embed_documents(
            documents, batch_size=100
        )

        # Add documents to storage provider
        if embedded_documents:
            logger.info("Storing documents in vector database")
            self.storage_provider.add(embedded_documents)
        else:
            logger.warning("No embedded documents produced by the pipeline")

        # Alert sources that we have updated
        for source in self.sources:
            if isinstance(source, HasPostIndexUpdateHook):
                source.post_index_update(self)

    def search_sources(
        self,
        query: str,
        *,
        overfetch_multiplier: int | None = None,
        max_overfetch_iterations: int | None = None,
    ) -> "BaseStorageQuerySet":
        """Search the index and return a queryish object of results
        mapped back to original source objects.
        Args:
            query: The search query string
        Returns:
            ResultQuerySet instance for the search results
        """
        return self.query_handler.search_sources(
            query,
            overfetch_multiplier=overfetch_multiplier,
            max_overfetch_iterations=max_overfetch_iterations,
        )

    def search_documents(self, query: str) -> "BaseStorageQuerySet":
        """Search the index and return a queryish object of Documents
        as stored in the underlying index.
        Args:
            query: The search query string
        Returns:
            ResultQuerySet instance for the search results
        """
        return self.query_handler.search_documents(query)

    def find_similar(self, obj: object) -> "BaseStorageQuerySet":
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
