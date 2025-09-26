import pytest

from django_ai_core.contrib.index.schema import EmbeddedDocument
from django_ai_core.contrib.index.storage.pgvector import PgVectorProvider
from django_ai_core.contrib.index.storage.pgvector.models import (
    PgVectorEmbedding,
    BasePgVectorEmbedding,
)

pytestmark = pytest.mark.django_db


def create_embedded_document(key="test:1", content="Test content", metadata=None):
    """Helper to create an embedded document for testing."""
    if metadata is None:
        metadata = {"source": "test"}

    return EmbeddedDocument(
        document_key=key, content=content, metadata=metadata, vector=[0.1, 0.2, 0.3]
    )


@pytest.fixture
def pg_vector_provider():
    return PgVectorProvider(index_name="test-index")


class TestPgVectorProvider:
    """Tests for the PgVectorProvider class."""

    def test_initialization(self, pg_vector_provider):
        """Test that provider initializes with correct attributes."""
        assert pg_vector_provider.index_name == "test-index"
        assert pg_vector_provider.model == PgVectorEmbedding

    def test_initialization_with_custom_model(self):
        """Test initialization with a custom model."""
        custom_model = type(
            "CustomPgVectorEmbedding",
            (BasePgVectorEmbedding,),
            {"vector": [0.1, 0.2, 0.3], "__module__": "testapp.models"},
        )

        provider = PgVectorProvider(model=custom_model, index_name="test-index")
        assert provider.model == custom_model
        assert provider.index_name == "test-index"

    def test_initialization_with_invalid_model(self):
        """Test initialization with an invalid model raises ValueError."""
        invalid_model = type("InvalidModel", (object,), {})

        with pytest.raises(ValueError) as excinfo:
            PgVectorProvider(model=invalid_model)

        assert "must include" in str(excinfo.value)

    def test_add_new_document(self, pg_vector_provider):
        """Test adding a new document to storage."""
        doc = create_embedded_document(key="test:1", content="Document 1")
        pg_vector_provider.add([doc])

        assert PgVectorEmbedding.objects.count() == 1
        assert PgVectorEmbedding.objects.get(pk="test:1")

    def test_update_existing_document(self, pg_vector_provider):
        """Test updating an existing document."""
        doc_original = create_embedded_document(key="test:1", content="Document 1")
        doc_updated = create_embedded_document(
            key="test:1", content="Updated Document 1"
        )
        pg_vector_provider.add([doc_original])
        instance = PgVectorEmbedding.objects.get(pk="test:1")
        assert instance.content == "Document 1"

        pg_vector_provider.add([doc_updated])

        updated_instance = PgVectorEmbedding.objects.get(pk="test:1")
        assert updated_instance.content == "Updated Document 1"

    def test_delete_documents(self, pg_vector_provider):
        """Test deleting documents by their keys."""
        docs = [
            create_embedded_document(key="test:1"),
            create_embedded_document(key="test:2"),
        ]
        pg_vector_provider.add(docs)
        pg_vector_provider.delete(["test:1"])

        assert PgVectorEmbedding.objects.filter(pk="test:1").count() == 0

    def test_clear(self, pg_vector_provider):
        """Test clearing all documents."""
        docs = [
            create_embedded_document(key="test:1"),
            create_embedded_document(key="test:2"),
        ]
        pg_vector_provider.add(docs)
        pg_vector_provider.clear()
        assert PgVectorEmbedding.objects.count() == 0

    def test_queryset_run_query(self, pg_vector_provider):
        """Test that PgVectorQuerySet.run_query works correctly."""
        doc1 = create_embedded_document(key="test:1", content="Document 1")
        doc2 = create_embedded_document(key="test:2", content="Document 2")
        pg_vector_provider.add([doc1, doc2])

        embedding = [0.7, 0.8, 0.9]
        queryset = pg_vector_provider.objects.filter(embedding=embedding)

        results = list(queryset)

        assert len(results) == 2
        assert results[0].document_key == "test:1"
        assert results[1].document_key == "test:2"

    def test_queryset_run_query_with_metadata_filter(self, pg_vector_provider):
        """Test querying with metadata filters."""
        # Create some test data
        doc1 = create_embedded_document(
            key="test:1", metadata={"category": "book", "author": "Test Author"}
        )
        doc2 = create_embedded_document(
            key="test:2", metadata={"category": "article", "author": "Another Author"}
        )
        pg_vector_provider.add([doc1, doc2])

        # Create a query with embedding and metadata filter
        embedding = [0.7, 0.8, 0.9]
        queryset = pg_vector_provider.objects.filter(
            embedding=embedding, category="book"
        )

        results = list(queryset)
        assert len(results) == 1
        assert results[0].document_key == "test:1"

    def test_queryset_run_query_without_embedding(self, pg_vector_provider):
        """Test that run_query raises ValueError if embedding is not provided."""
        queryset = pg_vector_provider.objects.filter(category="book")

        # Execute the query - should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            list(queryset)

        assert "embedding filter is required" in str(excinfo.value)

    def test_queryset_with_ordering_raises(self, pg_vector_provider):
        """Test that using ordering raises NotImplementedError."""
        queryset = pg_vector_provider.objects.filter(
            embedding=[0.1, 0.2, 0.3]
        ).order_by("field")

        with pytest.raises(NotImplementedError) as excinfo:
            list(queryset)

        assert "Ordering is not supported" in str(excinfo.value)

    def test_document_class_generation(self, pg_vector_provider):
        """Test document_cls property generates correct class."""
        doc_class = pg_vector_provider.document_cls

        assert doc_class.__name__ == "PgVectorDocument"
