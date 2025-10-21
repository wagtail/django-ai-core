from ..schema import EmbeddedDocument
from .base import BaseStorageQuerySet, StorageProvider


class InMemoryQuerySet(BaseStorageQuerySet["InMemoryProvider"]):
    def run_query(self):
        import numpy as np

        TEMP_FIXED_SIMILARITY_THRESHOLD = 0.2

        storage_provider = self.storage_provider
        filter_map = {filter[0]: filter[1] for filter in self.filters}

        embedding = filter_map.pop("embedding", None)
        if embedding is None:
            raise ValueError("embedding filter is required")

        similarities = []
        for document in storage_provider.documents.values():
            cosine_similarity = (
                np.dot(embedding, document.vector)
                / np.linalg.norm(embedding)
                * np.linalg.norm(document.vector)
            )
            if cosine_similarity >= TEMP_FIXED_SIMILARITY_THRESHOLD:
                similarities.append((cosine_similarity, document))

        sorted_similarities = sorted(
            similarities, key=lambda pair: pair[0], reverse=True
        )
        for document in [pair[1] for pair in sorted_similarities][
            self.start : self.stop
        ]:
            yield document


class InMemoryProvider(StorageProvider):
    """Simple in-memory storage for testing."""

    base_queryset_cls = InMemoryQuerySet

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.documents: dict[str, "EmbeddedDocument"] = {}

    def add(self, documents: list["EmbeddedDocument"]):
        """Store documents in memory."""
        for document in documents:
            self.documents[document.document_key] = document

    def delete(self, document_keys: list[str]):
        """Delete documents by their keys."""
        for key in document_keys:
            self.documents.pop(key, None)

    def clear(self):
        """Clear the vector database."""
        self.documents.clear()
