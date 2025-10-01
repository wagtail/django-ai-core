from typing import TYPE_CHECKING

from ..schema import EmbeddedDocument
from .base import BaseStorageDocument, BaseStorageQuerySet, StorageProvider

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import BasePydanticVectorStore


class LlamaIndexQuerySet(BaseStorageQuerySet["LlamaIndexProvider"]):
    def get_instance(self, val) -> BaseStorageDocument:
        if self.model:
            return self.model(
                document_key=val.document_key,
                content=val.content,
                metadata=val.metadata,
            )
        else:
            return val

    def run_query(self):
        from llama_index.core.vector_stores import MetadataFilters, VectorStoreQuery

        if not self.storage_provider:
            raise ValueError("Storage provider is required")

        storage_provider = self.storage_provider

        filter_map = {filter[0]: filter[1] for filter in self.filters}

        embedding = filter_map.pop("embedding", None)
        if embedding is None:
            raise ValueError("embedding filter is required")

        if self.ordering:
            raise NotImplementedError("Ordering is not supported for querying")

        if self.offset:
            raise NotImplementedError(
                "Offsets are not supported for the Llamaindex provider"
            )

        metadata_filters = MetadataFilters.from_dict(filter_map)

        query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=self.limit,
            filters=metadata_filters,
        )
        if storage_provider.vector_store is None:
            raise ValueError("Vector store is required")

        response = storage_provider.vector_store.query(query)

        if not response or not response.nodes:
            return

        for node in response.nodes:
            yield self.get_instance(node)


class LlamaIndexProvider(StorageProvider):
    """Vector storage using LlamaIndex vector stores."""

    base_queryset_cls = LlamaIndexQuerySet

    def __init__(
        self, *, vector_store: "BasePydanticVectorStore | None" = None, **kwargs
    ):
        super().__init__(**kwargs)
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
