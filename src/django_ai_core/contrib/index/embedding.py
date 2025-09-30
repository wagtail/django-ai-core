from abc import ABC, abstractmethod
from typing import Iterable

from django_ai_core.llm import LLMService

from .schema import Document, EmbeddedDocument


class EmbeddingTransformer(ABC):
    """Base class for embedding transformers which turn Documents into EmbeddedDocuments."""

    @property
    def transformer_id(self) -> str:
        """Get unique identifier for this transformer."""
        return self.__class__.__name__

    @abstractmethod
    def embed_string(self, text: str) -> list[float] | None:
        """Embed a string using the transformer."""
        pass

    @abstractmethod
    def embed_documents(
        self, documents: Iterable["Document"], *, batch_size: int = 100
    ) -> Iterable["EmbeddedDocument"]:
        """Add embeddings to multiple documents efficiently."""
        pass


class CoreEmbeddingTransformer(EmbeddingTransformer):
    """Embedding transformer that uses the core embeddings API."""

    def __init__(self, llm_service: LLMService):
        """Initialize with a core LLM Service instance.

        Args:
            llm_service: The LLM service
        """
        self.llm_service = llm_service

    @property
    def transformer_id(self) -> str:
        """Get unique identifier for this transformer."""
        return f"core_{self.llm_service.service_id}"

    def embed_string(self, text: str) -> list[float] | None:
        """Embed a string using the core embedding API."""
        return self.llm_service.embedding(text).data[0].embedding

    def embed_documents(
        self, documents: list["Document"], *, batch_size: int = 100
    ) -> list["EmbeddedDocument"]:
        """Add embedding vectors to multiple documents using the core embedding API.

        Args:
            documents: List of documents to transform
            batch_size: Number of documents to embed in each batch

        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return []

        embedded_documents = []

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [document.content for document in batch]
            embeddings = self.llm_service.embedding(texts).data

            for document, embedding in zip(batch, embeddings, strict=False):
                embedded_documents.append(document.add_embedding(embedding.embedding))

        return embedded_documents
