from unittest import mock

import pytest

from django_ai_core.contrib.index.embedding import EmbeddingTransformer
from django_ai_core.contrib.index.query import (
    DocumentResultMixin,
    QueryHandler,
    SourceResultMixin,
)
from django_ai_core.contrib.index.schema import Document
from django_ai_core.contrib.index.source import ObjectSource, Source
from django_ai_core.contrib.index.storage.base import (
    BaseStorageQuerySet,
    StorageProvider,
)


class MockObject:
    """Mock object that behaves like a Django model with pk-based equality."""

    def __init__(self, pk):
        self.pk = pk

    def __eq__(self, other):
        if isinstance(other, MockObject):
            return self.pk == other.pk
        return False

    def __hash__(self):
        return hash(self.pk)

    def __repr__(self):
        return f"MockObject(pk={self.pk})"


class MockEmbeddingTransformer(EmbeddingTransformer):
    """Mock embedding transformer for testing."""

    def embed_string(self, text):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, documents, batch_size=100):
        return [doc.add_embedding([0.1, 0.2, 0.3]) for doc in documents]


class MockObjectSource(ObjectSource):
    """Mock object source."""

    def __init__(self, source_id="test-source", objects=None):
        self._source_id = source_id
        self._objects = objects or []

    @property
    def source_id(self):
        return self._source_id

    def get_documents(self):
        return []

    def provides_document(self, document):
        return document.document_key.startswith(f"{self.source_id}:")

    def objects_to_documents(self, obj):
        return []

    def objects_from_documents(self, documents):
        for doc in documents:
            if self.provides_document(doc):
                pk = doc.metadata.get("pk")
                obj = MockObject(pk)
                yield obj

    def provides_object(self, obj):
        return obj in self._objects


class MockSource(Source):
    """Basic mock source for non-object sources."""

    def __init__(self, source_id="basic-source"):
        self._source_id = source_id

    @property
    def source_id(self):
        return self._source_id

    def get_documents(self):
        return []

    def provides_document(self, document):
        return document.document_key.startswith(f"{self.source_id}:")


class MockStorageQuerySet(BaseStorageQuerySet):
    """Mock storage provider queryset that lets us specify what Documents are returned."""

    def __init__(self, documents=None):
        super().__init__()
        self._documents = documents or []

    def run_query(self):
        """Mock run_query that returns documents with respect to limit."""
        limit = self.limit or len(self._documents)
        yield from self._documents[:limit]


class MockStorageProvider(StorageProvider):
    """Mock storage provider for testing."""

    def __init__(self, documents=None):
        super().__init__()
        self._documents = documents or []
        self._mock_queryset = MockStorageQuerySet(self._documents)

    @property
    def objects(self):
        return self._mock_queryset

    def add(self, documents):
        pass

    def delete(self, document_keys):
        pass

    def clear(self):
        pass


class TestDocumentResultMixin:
    def test_returns_raw_documents(self):
        """DocumentResultMixin should return raw documents without deduplication."""
        documents = [
            Document(
                document_key="test-source:obj1:chunk1",
                content="Content 1",
                metadata={"pk": 1, "chunk": 1},
            ),
            Document(
                document_key="test-source:obj1:chunk2",
                content="Content 2",
                metadata={"pk": 1, "chunk": 2},
            ),
            Document(
                document_key="test-source:obj2:chunk1",
                content="Content 3",
                metadata={"pk": 2, "chunk": 1},
            ),
        ]

        source = MockObjectSource()

        DocumentResultQuerySet = type(
            "DocumentResultQuerySet",
            (DocumentResultMixin, MockStorageQuerySet),
            {"sources": [source]},
        )

        qs = DocumentResultQuerySet(documents)
        results = list(qs.run_query())

        assert len(results) == 3
        assert results[0].metadata["pk"] == 1
        assert results[1].metadata["pk"] == 1
        assert results[2].metadata["pk"] == 2


class TestSourceResultMixin:
    @pytest.fixture(autouse=True)
    def setup(self):
        def make_qs(
            documents,
            *,
            source=None,
            overfetch_multiplier=None,
            max_overfetch_iterations=None,
        ):
            if not source:
                source = MockObjectSource()

            attrs = {"sources": [source]}
            if overfetch_multiplier:
                attrs["overfetch_multiplier"] = overfetch_multiplier
            if max_overfetch_iterations:
                attrs["max_overfetch_iterations"] = max_overfetch_iterations

            SourceResultQuerySet = type(
                "SourceResultQuerySet",
                (SourceResultMixin, MockStorageQuerySet),
                attrs,
            )
            return SourceResultQuerySet(documents)

        self.make_qs = make_qs

    def test_deduplicates_by_source(self):
        """SourceResultMixin should deduplicate documents to unique source objects."""
        documents = [
            Document(
                document_key="test-source:obj1:chunk1",
                content="Content 1",
                metadata={"pk": 1, "chunk": 1},
            ),
            Document(
                document_key="test-source:obj1:chunk2",
                content="Content 2",
                metadata={"pk": 1, "chunk": 2},
            ),
            Document(
                document_key="test-source:obj2:chunk1",
                content="Content 3",
                metadata={"pk": 2, "chunk": 1},
            ),
        ]
        qs = self.make_qs(documents)

        qs = qs[:20]
        results = list(qs.run_query())

        assert len(results) == 2
        result_pks = {obj.pk for obj in results}
        assert result_pks == {1, 2}

    def test_overfetch_no_offset(self):
        """SourceResultMixin should overfetch."""
        documents = [
            Document(
                document_key="test-source:1:1",
                content="c1",
                metadata={"pk": 1, "chunk": 1},
            ),
            Document(
                document_key="test-source:1:2",
                content="c2",
                metadata={"pk": 1, "chunk": 2},
            ),
            Document(
                document_key="test-source:1:3",
                content="c3",
                metadata={"pk": 1, "chunk": 3},
            ),
            Document(
                document_key="test-source:2:1",
                content="c4",
                metadata={"pk": 2, "chunk": 1},
            ),
            Document(
                document_key="test-source:3:1",
                content="c5",
                metadata={"pk": 3, "chunk": 1},
            ),
        ]

        qs = self.make_qs(documents)
        qs = qs[:2]

        results = list(qs.run_query())

        assert len(results) == 2

    def test_overfetch_with_offset(self):
        """SourceResultMixin should handle offset queries by fetching large batch."""
        documents = [
            Document(
                document_key="test-source:1:1",
                content="c1",
                metadata={"pk": 1, "chunk": 1},
            ),
            Document(
                document_key="test-source:2:1",
                content="c2",
                metadata={"pk": 2, "chunk": 1},
            ),
            Document(
                document_key="test-source:3:1",
                content="c3",
                metadata={"pk": 3, "chunk": 1},
            ),
            Document(
                document_key="test-source:4:1",
                content="c4",
                metadata={"pk": 4, "chunk": 1},
            ),
            Document(
                document_key="test-source:5:1",
                content="c5",
                metadata={"pk": 5, "chunk": 1},
            ),
        ]

        qs = self.make_qs(documents)
        qs = qs[2:4]

        results = list(qs.run_query())

        # Should return 2 results starting from offset 2 (pk=3, pk=4)
        assert len(results) == 2
        result_pks = [obj.pk for obj in results]
        assert result_pks == [3, 4]

    def test_respects_max_iterations(self):
        """SourceResultMixin should stop after max_overfetch_iterations."""
        # Many duplicates to force multiple iterations
        documents = [
            Document(
                document_key="test-source:1:1",
                content="c",
                metadata={"pk": 1, "chunk": i},
            )
            for i in range(20)
        ]

        qs = self.make_qs(documents, overfetch_multiplier=1, max_overfetch_iterations=2)
        qs = qs[0:10]
        qs._fetch_batch = mock.Mock(return_value=[documents[0]])

        list(qs.run_query())

        assert qs._fetch_batch.call_count == 2

    def test_handles_empty_results(self):
        """SourceResultMixin should handle empty document list gracefully."""
        qs = self.make_qs([])
        results = list(qs.run_query())

        assert len(results) == 0

    def test_documents_to_sources_with_non_object_source(self):
        """_documents_to_sources should return documents for non-ObjectSource."""
        documents = [
            Document(
                document_key="basic-source:doc1",
                content="Content 1",
                metadata={"id": 1},
            ),
        ]

        source = MockSource()

        qs = self.make_qs(documents, source=source)

        results = list(qs._documents_to_sources(documents))

        # Should return the document itself since it's not an ObjectSource
        assert len(results) == 1
        assert results[0] == documents[0]

    def test_custom_overfetch_multiplier(self):
        """Custom overfetch_multiplier should be respected."""
        documents = [
            Document(
                document_key="test-source:1:1",
                content="c",
                metadata={"pk": i, "chunk": 1},
            )
            for i in range(10)
        ]

        qs = self.make_qs(documents, overfetch_multiplier=5)
        qs = qs[:2]
        qs._fetch_batch = mock.Mock(return_value=[])

        list(qs.run_query())

        # First iteration should fetch limit * 5 = 10
        qs._fetch_batch.assert_called_once_with(limit=10)


class TestQueryHandler:
    @pytest.fixture(autouse=True)
    def setup(self):
        def make_handler(
            storage_provider=None, sources=None, embedding_transformer=None
        ):
            handler = QueryHandler()
            handler.configure(
                storage_provider=storage_provider or MockStorageProvider(),
                sources=sources or [MockObjectSource()],
                embedding_transformer=embedding_transformer
                or MockEmbeddingTransformer(),
            )
            return handler

        self.make_handler = make_handler

    def test_search_documents_returns_document_mixin(self):
        """search_documents should return queryset with DocumentResultMixin."""
        result = self.make_handler().search_documents("test query")

        # Check that the result class has DocumentResultMixin in its MRO
        assert DocumentResultMixin in result.__class__.__mro__

    def test_search_sources_returns_source_mixin(self):
        """search_sources should return queryset with SourceResultMixin."""
        result = self.make_handler().search_sources("test query")

        # Check that the result class has SourceResultMixin in its MRO
        assert SourceResultMixin in result.__class__.__mro__

    def test_search_sources_passes_overfetch_params(self):
        """search_sources should pass overfetch parameters to the queryset class."""
        result = self.make_handler().search_sources(
            "test query", overfetch_multiplier=5, max_overfetch_iterations=10
        )

        # Check that the queryset class has the custom parameters
        assert result.overfetch_multiplier == 5  # type: ignore
        assert result.max_overfetch_iterations == 10  # type: ignore

    def test_search_sources_uses_default_overfetch_params(self):
        """search_sources should use default overfetch parameters when not provided."""
        result = self.make_handler().search_sources("test query")

        # Check that the queryset class has default parameters from SourceResultMixin
        assert result.overfetch_multiplier == 3  # type: ignore
        assert result.max_overfetch_iterations == 3  # type: ignore

    def test_search_documents_raises_on_empty_query(self):
        """search_documents should raise ValueError for empty query."""
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            self.make_handler().search_documents("")

    def test_search_sources_raises_on_empty_query(self):
        """search_sources should raise ValueError for empty query."""
        with pytest.raises(ValueError, match="Search query cannot be empty"):
            self.make_handler().search_sources("")

    def test_search_documents_calls_embed_string(self):
        """search_documents should call embed_string with query."""
        embedding_transformer = MockEmbeddingTransformer()
        embedding_transformer.embed_string = mock.MagicMock(
            return_value=[0.1, 0.2, 0.3]
        )
        handler = self.make_handler(embedding_transformer=embedding_transformer)

        handler.search_documents("test query")

        embedding_transformer.embed_string.assert_called_once_with("test query")

    def test_search_sources_calls_embed_string(self):
        """search_sources should call embed_string with query."""
        embedding_transformer = MockEmbeddingTransformer()
        embedding_transformer.embed_string = mock.MagicMock(
            return_value=[0.1, 0.2, 0.3]
        )
        handler = self.make_handler(embedding_transformer=embedding_transformer)

        handler.search_sources("test query")

        embedding_transformer.embed_string.assert_called_once_with("test query")
