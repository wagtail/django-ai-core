"""
Query interface for vector indexes.
"""

from typing import TypeVar
from collections import defaultdict
import logging


from .source import Source
from .storage import BaseStorageQuerySet

logger = logging.getLogger(__name__)
ObjectType = TypeVar("ObjectType")


class ResultQuerySetMixin:
    """Wraps storage-specific Queryish QuerySets by delegating query modifiers to the inner queryset,
    but returns the original object rather than StorageDocuments"""

    sources: list[Source]

    def _get_objects_from_sources(self, results):
        sources_by_id = {source.source_id: source for source in self.sources}
        source_doc_mapping = defaultdict(list)
        for document in results:
            for source in self.sources:
                if source.document_belongs_to_source(document.document_key):
                    source_doc_mapping[source.source_id].append(document)

        objects = []
        for source_id, docs in source_doc_mapping.items():
            source = sources_by_id[source_id]
            objects.extend(source.get_objects_from_documents(docs))

        return objects

    def run_query(self):
        results = super().run_query()  # noqa
        yield from self._get_objects_from_sources(results)


class QueryHandler:
    """High-level query interface for vector indexes with Queryish support."""

    def configure(self, *, storage_provider, sources, embedding_transformer):
        """Configure the query handler with index components."""

        self.storage_provider = storage_provider
        self.sources = sources
        self.embedding_transformer = embedding_transformer

    def _build_result_query_set_cls(self):
        """Dynamically generate a new Queryish Queryset class from the storage provider queryset, but
        with the ResultQuerySetMixin that returns the source objects rather than Documents"""
        return type(
            "ResultQuerySet",
            (ResultQuerySetMixin, self.storage_provider.objects.__class__),
            {"sources": self.sources},
        )

    def search(
        self,
        query: str,
    ) -> ResultQuerySetMixin:
        """Search the index and return a queryish object of results.

        Args:
            query: The search query string
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = self.embedding_transformer.embed_string(query)
        queryset_cls = self._build_result_query_set_cls()

        return queryset_cls().filter(embedding=query_embedding)

    def find_similar(self, obj: object | Source) -> ResultQuerySetMixin:
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
        queryset_cls = self._build_result_query_set_cls()
        return queryset_cls().filter(embedding=query_embedding)
