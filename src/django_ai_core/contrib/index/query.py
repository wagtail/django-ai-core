import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable, TypeVar

from .source import ObjectSource, Source

logger = logging.getLogger(__name__)
ObjectType = TypeVar("ObjectType")

if TYPE_CHECKING:
    from .storage import BaseStorageQuerySet, StorageProvider


class BaseResultMixin:
    sources: list["Source"]
    storage_provider: "StorageProvider"

    @classmethod
    def build(
        cls,
        *,
        sources: list[Source],
        storage_provider: "StorageProvider",
        **extra_attrs,
    ):
        attrs = {
            "sources": sources,
            "storage_provider": storage_provider,
            **extra_attrs,
        }
        return type(
            "ResultQuerySet",
            (cls, storage_provider.objects.__class__),
            attrs,
        )


class DocumentResultMixin(BaseResultMixin):
    """
    Mixin for document search - returns raw documents without deduplication.

    This mixin doesn't override run_query(), so it inherits the storage
    provider's run_query() implementation, which returns raw documents.
    """

    def as_sources(self):
        qs = SourceResultMixin.build(
            sources=self.sources, storage_provider=self.storage_provider
        )()
        qs._results = self._results  # type: ignore
        qs._count = self._count  # type: ignore
        qs.filters = self.filters.copy()  # type: ignore
        return qs


class SourceResultMixin(BaseResultMixin):
    """
    Mixin for source search - returns unique source objects with over-fetching.

    This mixin overrides run_query() to implement over-fetching
    and deduplication. By the time run_query() is called, self.start and
    self.stop have been set by slicing operations.
    """

    overfetch_multiplier: int = 3
    max_overfetch_iterations: int = 3
    aggregation: str = "max"

    def as_documents(self):
        qs = DocumentResultMixin.build(
            sources=self.sources, storage_provider=self.storage_provider
        )()
        qs._results = self._results  # type: ignore
        qs._count = self._count  # type: ignore
        qs.filters = self.filters.copy()  # type: ignore
        return qs

    def run_query(self):
        """
        Execute query with over-fetching and deduplication.
        """
        requested_limit = self.limit
        requested_offset = self.offset or 0

        yield from self._overfetch(requested_offset, requested_limit)

    def _overfetch(self, requested_offset: int, requested_limit: int):
        """
        Over-fetch documents.

        This method fetches documents in batches, tracking unique sources,
        and continues fetching until we have enough unique sources or
        hit max_iterations.
        """
        all_objects = {}
        iteration = 0
        total_limit = requested_offset + requested_limit

        while (
            len(all_objects) < total_limit and iteration < self.max_overfetch_iterations
        ):
            fetch_size = total_limit * self.overfetch_multiplier * (iteration + 1)

            batch_docs = self._fetch_batch(limit=fetch_size)

            if not batch_docs:
                break

            all_objects.update(
                (obj, None) for obj in self._documents_to_sources(batch_docs)
            )

            iteration += 1

            if len(batch_docs) < fetch_size:
                # We got fewer docs than expected so there's no need to keep iterating
                break

        yield from itertools.islice(all_objects, requested_offset, total_limit)

    def _fetch_batch(self, limit: int):
        """
        Fetch a batch of documents from storage provider.

        This method calls the parent class's (storage provider's) run_query()
        with specific offset and limit.
        """
        temp_qs = self.clone(start=0, stop=limit)  # type: ignore

        # Skip SourceResultMixin in the MRO of temp_qs to resolve
        # run_query to the method on the storage provided queryish object
        batch_docs = list(super(SourceResultMixin, temp_qs).run_query())  # type: ignore

        return batch_docs

    def _documents_to_sources(self, documents) -> Iterable[Any]:
        """
        Convert documents to their original source objects.
        """
        if not documents:
            return

        sources_by_id = {source.source_id: source for source in self.sources}
        source_doc_mapping = defaultdict(list)
        mapped_objects = {}

        # Iterate once to to find all objects belonging to a source
        for document in documents:
            for source in self.sources:
                if source.provides_document(document):
                    source_doc_mapping[source.source_id].append(document)
                    break

        # then iterate through those bundles of objects as doing bulk
        # conversion is more efficient
        for source_id, docs in source_doc_mapping.items():
            source = sources_by_id[source_id]
            document_keys = [doc.document_key for doc in docs]
            if isinstance(source, ObjectSource):
                objects = source.objects_from_documents(docs)
                mapped_objects.update(zip(document_keys, objects, strict=True))
            else:
                mapped_objects.update(zip(document_keys, docs, strict=True))

        for document in documents:
            yield mapped_objects[document.document_key]


class QueryHandler:
    """High-level query interface for vector indexes with Queryish support."""

    def configure(self, *, storage_provider, sources, embedding_transformer):
        """Configure the query handler with index components."""

        self.storage_provider = storage_provider
        self.sources: list[Source] = sources
        self.embedding_transformer = embedding_transformer

    def search_sources(
        self,
        query: str,
        *,
        overfetch_multiplier: int | None = None,
        max_overfetch_iterations: int | None = None,
    ) -> "BaseStorageQuerySet":
        """Search the index and return a queryish object of results
        mapped back to their original source objects.

        Args:
            query: The search query string
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = self.embedding_transformer.embed_string(query)
        queryset_attrs = {
            k: v
            for k, v in {
                "overfetch_multiplier": overfetch_multiplier,
                "max_overfetch_iterations": max_overfetch_iterations,
            }.items()
            if v is not None
        }
        queryset_cls = SourceResultMixin.build(
            sources=self.sources,
            storage_provider=self.storage_provider,
            **queryset_attrs,
        )

        return queryset_cls().filter(embedding=query_embedding)

    def search_documents(
        self,
        query: str,
    ) -> "BaseStorageQuerySet":
        """Search the index and return a queryish object of Documents
        as returned from the storage provider.

        Args:
            query: The search query string
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        query_embedding = self.embedding_transformer.embed_string(query)
        queryset_cls = DocumentResultMixin.build(
            sources=self.sources,
            storage_provider=self.storage_provider,
        )

        return queryset_cls().filter(embedding=query_embedding)

    def find_similar(self, obj: object) -> "BaseStorageQuerySet":
        """Find objects similar to the given object.

        Args:
            obj: The object to find similar objects to.

        Raises:
            ValueError: If no source is found for the object
        """
        # Try to find an appropriate source
        for source in self.sources:
            if isinstance(source, ObjectSource) and source.provides_object(obj):
                source_to_use = source
                break

        if not source_to_use:
            raise ValueError(
                "No suitable source found for query object. The object must belong to one of the object sources configured on the index, or a source must be provided."
            )

        documents = source_to_use.objects_to_documents(obj)
        embedded_documents = self.embedding_transformer.embed_documents(documents)
        # Just use the first document as the query embedding
        query_embedding = embedded_documents[0].vector
        queryset_cls = SourceResultMixin.build(
            sources=self.sources, storage_provider=self.storage_provider
        )
        return queryset_cls().filter(embedding=query_embedding)
