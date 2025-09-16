"""
Query interface for vector indexes.
"""

from typing import TypeVar
import logging

from queryish import Queryish

from .schema import Document
from .source import Source
from .storage import StorageQuerySet

logger = logging.getLogger(__name__)
ObjectType = TypeVar("ObjectType")


class VectorQueryish(Queryish):
    """QuerySet-like interface for vector search results."""

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
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = self.embedding_transformer.embed_string(query)

        return self.storage_provider.search(query_embedding)

    def find_similar(self, obj: object | Source) -> StorageQuerySet:
        """Find objects similar to the given object.

        Args:
            obj: The object to find similar objects to, or a Source object

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
