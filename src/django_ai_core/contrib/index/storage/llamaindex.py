from typing import TYPE_CHECKING

from .base import StorageProvider, StorageQuerySet, StorageDocument
from ..schema import EmbeddedDocument

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import BasePydanticVectorStore


class LlamaIndexQuerySet(StorageQuerySet["LlamaIndexProvider"]):
    def get_instance(self, val) -> StorageDocument:
        if self.model:
            return self.model(
                document_key=val.document_key,
                content=val.content,
                metadata=val.metadata,
            )
        else:
            return val

    def run_query(self):
        from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilters

        if not self.storage_provider:
            raise ValueError("Storage provider is required")

        filter_map = {filter[0]: filter[1] for filter in self.filters}

        embedding = filter_map.pop("embedding", None)
        if embedding is None:
            raise ValueError("embedding filter is required")

        if self.ordering:
            raise NotImplementedError("Ordering is not supported for querying")

        metadata_filters = MetadataFilters.from_dict(filter_map)

        query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=self._top_k,
            filters=metadata_filters,
        )
        if self.storage_provider.vector_store is None:
            raise ValueError("Vector store is required")

        response = self.storage_provider.vector_store.query(query)

        if not response or not response.nodes:
            return

        for node in response.nodes:
            yield self.get_instance(node)


class LlamaIndexProvider(StorageProvider):
    """Vector storage using LlamaIndex vector stores."""

    queryset_cls = LlamaIndexQuerySet

    def __init__(self, vector_store: "BasePydanticVectorStore | None" = None):
        if vector_store is None:
            from llama_index.core.vector_stores.simple import SimpleVectorStore

            self.vector_store = SimpleVectorStore()
        else:
            self.vector_store = vector_store

    def add(self, documents: list["EmbeddedDocument"]):
        """Store documents in the vector store."""
        from llama_index.core import Document as LlamaDocument

        # Convert documents to LlamaIndex nodes
        nodes = []
        for document in documents:
            node = LlamaDocument(
                text=document.content,
                metadata=document.metadata,
                id_=document.document_key,
                embedding=document.vector,
            )
            nodes.append(node)

        # Create or update index
        self.vector_store.add(nodes)

    def delete(self, document_keys: list[str]):
        """Delete documents by their keys."""
        self.vector_store.delete_nodes(document_keys)

    def clear(self):
        """Clear the vector database."""
        self.vector_store.clear()
