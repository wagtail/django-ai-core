"""
Query interface for vector indexes.

This module provides the query interfaces for vector indexes, including
a Queryish implementation for vector search results.
"""

from typing import TypeVar, Iterable, Union
import logging
import time

from queryish import Queryish

from .schema import Document
from .source import Source
from .storage import StorageQuerySet

logger = logging.getLogger(__name__)
ObjectType = TypeVar("ObjectType")


class VectorDocument:
    """Wrapper for Document objects to provide a consistent interface for Queryish."""

    def __init__(self, document: Document, source: Source | None = None):
        self.document = document
        self.source = source
        self._object = None
        self._object_loaded = False

    @property
    def content(self) -> str:
        """Get the document content."""
        return self.document.content

    @property
    def document_key(self) -> str:
        """Get the document key."""
        return self.document.document_key

    @property
    def metadata(self) -> dict:
        """Get the document metadata."""
        return self.document.metadata

    @property
    def source_id(self) -> str | None:
        """Get the source ID from metadata."""
        return self.metadata.get("source_id")

    def get_object(self) -> object | None:
        """Get the original object for this document."""
        if not self._object_loaded and self.source is not None:
            self._object = self.source.get_objects_from_documents([self.document])
            self._object = self._object[0] if self._object else None
            self._object_loaded = True
        return self._object

    def __str__(self) -> str:
        return f"VectorDocument({self.document_key})"


class VectorQueryish(Queryish):
    """QuerySet-like interface for vector search results."""

    def __init__(
        self,
        vector_index=None,
        embedding_transformer=None,
        storage=None,
        sources=None,
        query_string=None,
        query_embedding=None,
        query_object=None,
        query_object_source=None,
        limit=10,
        offset=0,
        filters=None,
        ordering=None,
        allow_empty=True,
    ):
        """Initialize a vector search queryset.

        Args:
            vector_index: VectorIndex instance (optional if storage and embedding_transformer provided)
            embedding_transformer: Embedding transformer for vectorizing queries
            storage: Vector storage backend
            sources: List of sources for converting documents to objects
            query_string: Text query for similarity search
            query_embedding: Pre-computed query embedding
            query_object: Object to find similar items to
            limit: Maximum number of results to return
            offset: Offset for pagination
            filters: Dictionary of metadata filters
            ordering: List of fields to order by
            allow_empty: Whether to allow empty results
        """
        self.vector_index = vector_index

        # Get components from vector_index if not provided directly
        self.embedding_transformer = embedding_transformer or getattr(
            vector_index, "embedding_transformer", None
        )
        self.storage = storage or getattr(vector_index, "storage", None)
        self.sources = sources or getattr(vector_index, "sources", [])

        # Query parameters
        self.query_string = query_string
        self.query_embedding = query_embedding
        self.query_object = query_object

        # Configure queryset behavior
        self.limit = limit
        self.offset = offset
        self._filters = filters or {}
        self._ordering = ordering or []
        self.allow_empty = allow_empty

        # Results cache
        self._results = None
        self._count = None

        # Define filter and ordering fields
        self.filter_fields = {"source_id", "document_key"}
        self.ordering_fields = {"document_key", "content"}

        # Validate components
        if not self.storage_provider:
            raise ValueError("Vector storage is required")

        if not self.embedding_transformer and not self.query_embedding:
            raise ValueError(
                "Either embedding_transformer or query_embedding must be provided"
            )

        # Check that at least one query method is provided
        if not any([self.query_string, self.query_embedding, self.query_object]):
            if not allow_empty:
                raise ValueError(
                    "At least one of query_string, query_embedding, or query_object must be provided"
                )

    def filter(self, **kwargs) -> "VectorQueryish":
        """Filter the queryset by metadata fields."""
        clone = self.clone()

        # Add filters
        for key, value in kwargs.items():
            if key not in self.filter_fields:
                raise ValueError(f"Invalid filter field: {key}")
            clone._filters[key] = value

        # Reset cached results
        clone._results = None
        clone._count = None

        return clone

    def order_by(self, *fields) -> "VectorQueryish":
        """Order the queryset by the given fields."""
        clone = self.clone()

        # Validate fields
        for field in fields:
            # Strip '-' for descending
            plain_field = field[1:] if field.startswith("-") else field
            if plain_field not in self.ordering_fields:
                raise ValueError(f"Invalid ordering field: {plain_field}")

        clone._ordering = fields

        # Reset cached results
        clone._results = None

        return clone

    def __getitem__(self, key) -> Union["VectorQueryish", VectorDocument]:
        """Support slicing and indexing."""
        if isinstance(key, slice):
            clone = self.clone()

            # Calculate new offset and limit
            start = key.start or 0
            stop = key.stop if key.stop is not None else self.limit + self.offset

            # Apply relative to current offset/limit
            clone.offset = self.offset + start
            clone.limit = max(0, stop - start)

            # Reset cached results
            clone._results = None

            return clone
        elif isinstance(key, int):
            # For indexing, convert to 0-based index
            if key < 0:
                # Convert negative index to positive
                count = self.count()
                key = count + key
                if key < 0:
                    raise IndexError("Index out of range")

            # Check if key is within range
            if key >= self.limit:
                raise IndexError("Index out of range")

            # Get results
            results = list(self)
            if key >= len(results):
                raise IndexError("Index out of range")

            return results[key]
        else:
            raise TypeError("Invalid index or slice")

    def _prepare_query_embedding(self) -> list[float]:
        """Prepare the query embedding based on available inputs."""
        if self.query_embedding is not None:
            return self.query_embedding

        if self.query_string is not None:
            return self.embedding_transformer.embed_string(self.query_string)

        if self.query_object is not None:
            source = self.query_object_source
            if source is None:
                # Try to find an appropriate source
                for s in self.sources:
                    if (
                        hasattr(s, "model")
                        and hasattr(self.query_object, "_meta")
                        and s.model == self.query_object._meta.model
                    ):
                        source = s
                        break

            if source is None:
                raise ValueError("No suitable source found for query object")

            content = source.get_content(self.query_object)
            return self.embedding_transformer.embed_string(content)

        # If we get here, we have no query
        return None

    def _filter_documents(self, documents: list[Document]) -> list[Document]:
        """Filter documents by metadata filters."""
        if not self._filters:
            return documents

        filtered = []
        for document in documents:
            matches_all = True
            for key, value in self._filters.items():
                if key == "source_id":
                    if document.metadata.get("source_id") != value:
                        matches_all = False
                        break
                elif key == "document_key":
                    if document.document_key != value:
                        matches_all = False
                        break
                # Add other filter types as needed

            if matches_all:
                filtered.append(document)

        return filtered

    def _apply_ordering(self, documents: list[Document]) -> list[Document]:
        """Apply ordering to documents."""
        if not self._ordering:
            return documents

        # Sort by each field in reverse order
        for field in reversed(self._ordering):
            descending = field.startswith("-")
            field_name = field[1:] if descending else field

            if field_name == "document_key":
                documents.sort(key=lambda d: d.document_key, reverse=descending)
            elif field_name == "content":
                documents.sort(key=lambda d: d.content, reverse=descending)
            # Add other ordering fields as needed

        return documents

    def _fetch_documents(self) -> list[Document]:
        """Fetch documents from vector storage."""
        # Start timing
        start_time = time.time()

        # Get query embedding
        query_embedding = self._prepare_query_embedding()
        if query_embedding is None and not self.allow_empty:
            raise ValueError("No valid query provided")

        if query_embedding is None:
            # Return empty result for empty query
            return []

        # Query vector storage with extra results for filtering
        fetch_limit = (
            max(50, (self.limit + self.offset) * 3)
            if self._filters
            else (self.limit + self.offset)
        )
        documents = self.storage_provider.query(query_embedding, fetch_limit)

        # Apply filters
        filtered_documents = self._filter_documents(documents)

        # Apply ordering
        ordered_documents = self._apply_ordering(filtered_documents)

        # Store total available
        self.total_available = len(ordered_documents)

        # Apply pagination
        paginated_documents = ordered_documents[self.offset : self.offset + self.limit]

        # Calculate query time
        self.query_time_ms = (time.time() - start_time) * 1000

        # Update metadata
        self.metadata = {
            "query_time_ms": self.query_time_ms,
            "total_available": self.total_available,
            "offset": self.offset,
            "limit": self.limit,
            "filters": self._filters,
        }

        return paginated_documents

    def run_query(self) -> Iterable[VectorDocument]:
        """Execute the query and return the results.

        This is the main method that must be implemented for Queryish.
        """
        try:
            documents = self._fetch_documents()

            # Map sources by ID for efficient lookup
            source_map = {source.source_id: source for source in self.sources}

            # Wrap documents with source information
            result_documents = []
            for document in documents:
                source_id = document.metadata.get("source_id")
                source = source_map.get(source_id) if source_id else None
                result_documents.append(VectorDocument(document, source))

            return result_documents
        except Exception as e:
            logger.error(f"Error executing vector query: {e}")
            raise RuntimeError(f"Failed to execute vector query: {e}") from e

    def run_count(self) -> int:
        """Count the total number of results."""
        if self._count is not None:
            return self._count

        # If we have results already, use their length
        if self._results is not None:
            self._count = len(self._results)
            return self._count

        # Otherwise, fetch documents and count them
        try:
            documents = self._fetch_documents()
            self._count = len(documents)
            return self._count
        except Exception as e:
            logger.error(f"Error counting vector query results: {e}")
            raise RuntimeError(f"Failed to count vector query results: {e}") from e

    def get_objects(self) -> list[object]:
        """Convert documents to their original objects."""
        objects = []
        seen_keys = set()

        for vector_doc in self:
            obj = vector_doc.get_object()
            if obj is not None:
                # Create a unique key for deduplication
                if hasattr(obj, "_meta") and hasattr(obj, "pk"):
                    # Django model
                    obj_key = (obj._meta.label, obj.pk)
                elif hasattr(obj, "__dict__"):
                    # Generic object
                    obj_key = str(obj)
                else:
                    # Fallback
                    obj_key = id(obj)

                if obj_key not in seen_keys:
                    seen_keys.add(obj_key)
                    objects.append(obj)

        return objects


class QueryHandler:
    """High-level query interface for vector indexes with Queryish support."""

    def configure(self, *, storage_provider, sources, embedding_transformer):
        """Configure the query handler with index components."""

        self.storage_provider = storage_provider
        self.sources = sources
        self.embedding_transformer = embedding_transformer

    def search(
        self,
        query: str,
    ) -> StorageQuerySet:
        """Search the index and return a queryish object of results.

        Args:
            query: The search query string

        Returns:
            VectorQueryish instance for the search results

        Raises:
            ValueError: If the query is empty or invalid
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = self.embedding_transformer.embed_string(query)

        return self.storage_provider.search(query_embedding)

    def find_similar(self, obj: object | Source) -> StorageQuerySet:
        """Find objects similar to the given object.

        Args:
            obj: The object to find similar objects to, or a Source object

        Returns:
            VectorQueryish instance for the search results

        Raises:
            ValueError: If no source is found for the object
        """
        if isinstance(obj, Source):
            source_to_use = obj
        else:
            # Try to find an appropriate source
            for source in self.sources:
                if source.object_belongs_to_source(obj):
                    source_to_use = source
                    break

        if not source_to_use:
            raise ValueError(
                "No suitable source found for query object. The object must belong to one of the sources configured on the index, or a source must be provided."
            )

        documents = source_to_use.get_documents_for_object(obj)
        embedded_documents = self.embedding_transformer.transform(documents)
        # Just use the first document as the query embedding
        query_embedding = embedded_documents[0].vector
        return self.storage_provider.search(query_embedding)
