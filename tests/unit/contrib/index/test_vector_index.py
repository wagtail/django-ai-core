from typing import ClassVar
from unittest import mock

from django_ai_core.contrib.index.base import VectorIndex
from django_ai_core.contrib.index.schema import Document
from django_ai_core.contrib.index.storage.base import StorageProvider
from django_ai_core.contrib.index.embedding import EmbeddingTransformer
from django_ai_core.contrib.index.source import HasPostIndexUpdateHook


class MockStorageProvider(StorageProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.added_documents = []
        self.cleared = False
        self.deleted_keys = []
        self.base_queryset_cls = mock.MagicMock()

    def add(self, documents):
        self.added_documents.extend(documents)

    def delete(self, document_keys):
        self.deleted_keys.extend(document_keys)

    def clear(self):
        self.cleared = True
        self.added_documents = []


class MockEmbeddingTransformer(EmbeddingTransformer):
    def __init__(self):
        self.embedded_documents = []

    def embed_string(self, text):
        # Return a simple mock embedding
        return [0.1, 0.2, 0.3]

    def embed_documents(self, documents, batch_size=100):
        embedded = []
        for doc in documents:
            embedded_doc = doc.add_embedding([0.1, 0.2, 0.3])
            embedded.append(embedded_doc)
        self.embedded_documents.extend(embedded)
        return embedded


class MockSource(HasPostIndexUpdateHook):
    def __init__(self, documents=None):
        self.documents = documents or []
        self.post_update_called = False

    @property
    def source_id(self):
        return "mock-source"

    def get_documents(self):
        return self.documents

    def provides_document(self, document):
        return document.document_key.startswith(self.source_id)

    def post_index_update(self, index):
        self.post_update_called = True


class MockVectorIndex(VectorIndex):
    sources: ClassVar[list[MockSource]]
    storage_provider: ClassVar[MockStorageProvider]
    embedding_transformer: ClassVar[MockEmbeddingTransformer]


def create_test_vector_index_cls(
    sources=None, storage_provider=None, embedding_transformer=None
):
    return type(
        "TestVectorIndex",
        (MockVectorIndex,),
        {
            "sources": sources or [MockSource()],
            "storage_provider": storage_provider or MockStorageProvider(),
            "embedding_transformer": embedding_transformer
            or MockEmbeddingTransformer(),
        },
    )


def test_vector_index_build_empty():
    """Test building an index with no documents."""
    index = create_test_vector_index_cls()()
    result = index.build()

    assert result == index
    assert len(index.storage_provider.added_documents) == 0


def test_vector_index_build_with_documents():
    """Test building an index with documents."""
    documents = [
        Document(
            document_key="mock-source:1", content="Document 1", metadata={"id": 1}
        ),
        Document(
            document_key="mock-source:2", content="Document 2", metadata={"id": 2}
        ),
    ]
    source = MockSource(documents)

    index = create_test_vector_index_cls(sources=[source])()
    index.build()

    assert len(index.embedding_transformer.embedded_documents) == 2

    assert len(index.storage_provider.added_documents) == 2
    assert index.storage_provider.added_documents[0].document_key == "mock-source:1"
    assert index.storage_provider.added_documents[1].document_key == "mock-source:2"


def test_vector_index_update():
    """Test updating an index with new documents."""
    index = create_test_vector_index_cls()()

    documents = [
        Document(
            document_key="mock-source:1", content="Document 1", metadata={"id": 1}
        ),
        Document(
            document_key="mock-source:2", content="Document 2", metadata={"id": 2}
        ),
    ]

    index.update(documents)

    assert len(index.embedding_transformer.embedded_documents) == 2
    assert len(index.storage_provider.added_documents) == 2


def test_vector_index_post_update_hook():
    """Test that post_index_update hook is called for sources that support it."""
    index = create_test_vector_index_cls()()

    documents = [
        Document(
            document_key="mock-source:1", content="Document 1", metadata={"id": 1}
        ),
    ]
    index.sources[0].documents = documents
    index.build()

    assert index.sources[0].post_update_called


def test_vector_index_search():
    """Test search delegates to query_handler."""
    index = create_test_vector_index_cls()()

    index.query_handler.search = mock.MagicMock()

    index.search("test query")

    index.query_handler.search.assert_called_once_with("test query")


def test_vector_index_find_similar():
    """Test find_similar delegates to query_handler."""
    index = create_test_vector_index_cls()()

    index.query_handler.find_similar = mock.MagicMock()
    test_obj = object()

    index.find_similar(test_obj)

    index.query_handler.find_similar.assert_called_once_with(test_obj)
