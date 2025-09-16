"""
Schema definitions for vector indexing.

This module contains the core data structures used throughout the indexing system.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    """
    Represents a document to be indexed.

    Documents are created by splitting source objects into smaller, more manageable pieces
    for embedding and retrieval.
    """

    document_key: str
    content: str
    metadata: dict[str, Any]

    def add_embedding(self, embedding: list[float]) -> "EmbeddedDocument":
        """Create a new EmbeddedDocument with the given embedding."""
        return EmbeddedDocument(
            document_key=self.document_key,
            content=self.content,
            metadata=self.metadata,
            vector=embedding,
        )


@dataclass
class EmbeddedDocument(Document):
    """
    Represents a document with an associated vector embedding.

    This is the final form of content that gets stored in the vector database.
    """

    vector: list[float]
