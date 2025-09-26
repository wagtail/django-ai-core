from unittest import mock

from django_ai_core.contrib.index.schema import EmbeddedDocument
from django_ai_core.contrib.index.storage.inmemory import InMemoryProvider


def create_embedded_document(key="test:1", content="Test content", metadata=None):
    """Helper to create an embedded document for testing."""
    if metadata is None:
        metadata = {"source": "test"}

    return EmbeddedDocument(
        document_key=key, content=content, metadata=metadata, vector=[0.1, 0.2, 0.3]
    )


class TestInMemoryProvider:
    """Tests for the InMemoryProvider."""

    def test_initialization(self):
        """Test basic initialization."""
        provider = InMemoryProvider(index_name="test_index")
        assert provider.index_name == "test_index"
        assert provider.documents == {}

    def test_add_documents(self):
        """Test adding documents to storage."""
        provider = InMemoryProvider()

        doc1 = create_embedded_document(key="test:1", content="Document 1")
        doc2 = create_embedded_document(key="test:2", content="Document 2")

        provider.add([doc1, doc2])

        assert len(provider.documents) == 2
        assert provider.documents["test:1"] is doc1
        assert provider.documents["test:2"] is doc2

    def test_delete_documents(self):
        """Test deleting documents from storage."""
        provider = InMemoryProvider()

        doc1 = create_embedded_document(key="test:1")
        doc2 = create_embedded_document(key="test:2")
        provider.add([doc1, doc2])

        provider.delete(["test:1"])

        assert len(provider.documents) == 1
        assert "test:1" not in provider.documents
        assert "test:2" in provider.documents

    def test_clear(self):
        """Test clearing all documents."""
        provider = InMemoryProvider()

        provider.add(
            [
                create_embedded_document(key="test:1"),
                create_embedded_document(key="test:2"),
            ]
        )

        assert len(provider.documents) == 2
        provider.clear()
        assert len(provider.documents) == 0

    def test_document_class_generation(self):
        """Test document_cls property generates correct class."""
        provider = InMemoryProvider()
        doc_class = provider.document_cls

        assert doc_class.__name__ == "InMemoryDocument"

        assert doc_class.Meta.storage_provider is provider

    def test_objects_property(self):
        """Test objects property returns queryable interface."""
        provider = InMemoryProvider()

        query_builder = provider.objects
        assert hasattr(query_builder, "filter")

    @mock.patch("numpy.dot")
    @mock.patch("numpy.linalg.norm")
    def test_queryset_run_query(self, mock_norm, mock_dot):
        """Test InMemoryQuerySet run_query with mocked numpy."""
        provider = InMemoryProvider()

        mock_dot.return_value = 0.5
        mock_norm.return_value = 1.0

        doc = create_embedded_document(key="test:1")
        provider.add([doc])

        embedding = [0.4, 0.5, 0.6]
        queryset = provider.objects.filter(embedding=embedding)

        results = list(queryset)

        assert len(results) == 1
        assert results[0].document_key == "test:1"

        mock_dot.assert_called_with(embedding, doc.vector)
        mock_norm.assert_any_call(embedding)
        mock_norm.assert_any_call(doc.vector)
