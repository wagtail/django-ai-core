import boto3

from ..schema import EmbeddedDocument
from .base import BaseStorageDocument, BaseStorageQuerySet, StorageProvider

# Key used for storing original content in non-filterable metadata
CONTENT_METADATA_KEY = "dj_ai_core_content"


class S3VectorQuerySet(BaseStorageQuerySet["S3VectorProvider"]):
    def get_instance(self, val) -> BaseStorageDocument:
        if self.model:
            metadata = val["metadata"]
            content = metadata.pop(CONTENT_METADATA_KEY, "")
            return self.model(
                document_key=val["key"],
                content=content,
                metadata=metadata,
                score=1 - val["distance"],
            )
        else:
            return val

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
                "Offsets are not supported for the S3 Vectors provider"
            )

        response = client.query_vectors(
            vectorBucketName=storage_provider.bucket_name,
            indexName=storage_provider.index_name,
            topK=self.limit,
            queryVector={"float32": embedding},
            # Only supporting strict equality in filters for now
            filter=filter_map or None,
            returnMetadata=True,
            returnDistance=True,
        )

        for vector in response["vectors"]:
            yield self.get_instance(vector)


class S3VectorProvider(StorageProvider):
    """Vector storage using S3 vector stores."""

    base_queryset_cls = S3VectorQuerySet

    def __init__(
        self,
        *,
        bucket_name: str | None = None,
        dimensions: int,
        distance_metric: str = "cosine",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bucket_name = bucket_name
        self.dimensions = dimensions
        self.distance_metric = distance_metric

        self.client = boto3.client("s3vectors")

    @property
    def index_name(self):
        if not self._index_name:
            raise ValueError("S3 Vector Provider must have an index_name configured")
        return self._index_name

    @index_name.setter
    def index_name(self, value):
        if value:
            self._index_name = value.replace("_", "-")

    def _create_or_get_index(self):
        # TODO add logging
        try:
            self.client.get_index(
                vectorBucketName=self.bucket_name, indexName=self.index_name
            )
        except self.client.exceptions.NotFoundException:
            self.client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dataType="float32",
                dimension=self.dimensions,
                distanceMetric=self.distance_metric,
                metadataConfiguration={
                    "nonFilterableMetadataKeys": [CONTENT_METADATA_KEY]
                },
            )

    def add(self, documents: list["EmbeddedDocument"]):
        """Store documents in the vector store."""
        self._create_or_get_index()
        vectors = []
        for doc in documents:
            vectors.append(
                {
                    "key": doc.document_key,
                    "data": {"float32": doc.vector},
                    "metadata": {**doc.metadata, CONTENT_METADATA_KEY: doc.content},
                }
            )

        self.client.put_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            vectors=vectors,
        )

    def delete(self, document_keys: list[str]):
        """Delete documents by their keys."""
        self.client.delete_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            keys=document_keys,
        )

    def clear(self):
        """Clear the vector database."""
        self.client.delete_index(
            vectorBucketName=self.bucket_name, indexName=self.index_name
        )
