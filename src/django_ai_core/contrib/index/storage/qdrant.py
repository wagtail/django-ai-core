import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.models import Distance

from ..schema import EmbeddedDocument
from .base import BaseStorageDocument, BaseStorageQuerySet, StorageProvider

# Key used for storing original content in metadata
CONTENT_METADATA_KEY = "dj_ai_core_content"


class QdrantQuerySet(BaseStorageQuerySet["QdrantProvider"]):
    def get_instance(self, val) -> BaseStorageDocument:
        if self.model:
            metadata = val.payload
            content = metadata.pop(CONTENT_METADATA_KEY)
            return self.model(
                document_key=metadata["document_key"],
                content=content,
                metadata=metadata,
                score=val.score,
            )
        else:
            return val.payload

    def run_query(self):
        if not self.storage_provider:
            raise ValueError("Storage provider is required")

        storage_provider = self.storage_provider
        client = self.storage_provider.client

        filter_map = {filter[0]: filter[1] for filter in self.filters}

        embedding = filter_map.pop("embedding", None)
        if embedding is None:
            raise ValueError("embedding filter is required")

        if self.ordering:
            raise NotImplementedError("Ordering is not supported for querying")

        if self.offset:
            raise NotImplementedError(
                "Offsets are not supported for the Qdrant provider"
            )

        filters = [
            qdrant_models.FieldCondition(key=idx, match=val)
            for idx, val in filter_map.items()
        ]

        response = client.search(
            collection_name=storage_provider.index_name,
            query_vector=embedding,
            limit=self.limit,
            query_filter=qdrant_models.Filter(must=filters),
        )

        for vector in response:
            yield self.get_instance(vector)


class QdrantProvider(StorageProvider):
    """Vector storage using Qdrant."""

    base_queryset_cls = QdrantQuerySet

    def __init__(
        self,
        *,
        host: str,
        port: int = 6333,
        api_key: str,
        dimensions: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client = QdrantClient(url=host, port=port, api_key=api_key)
        self.dimensions = dimensions

    def _create_or_get_index(self):
        if not self.client.collection_exists(self.index_name):
            self.client.create_collection(
                collection_name=self.index_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.dimensions, distance=Distance.COSINE
                ),
            )

    def add(self, documents: list["EmbeddedDocument"]):
        """Store documents in the vector store."""
        self._create_or_get_index()
        points = []
        for doc in documents:
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=doc.vector,
                    payload={
                        **doc.metadata,
                        CONTENT_METADATA_KEY: doc.content,
                        "document_key": doc.document_key,
                    },
                )
            )

        self.client.upsert(collection_name=self.index_name, points=points)

    def delete(self, document_keys: list[str]):
        """Delete documents by their keys."""
        self.client.delete(
            collection_name=self.index_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_key",
                            match=qdrant_models.MatchAny(any=document_keys),
                        )
                    ]
                )
            ),
        )

    def clear(self):
        """Clear the vector database."""
        self.client.delete_collection(collection_name=self.index_name)
