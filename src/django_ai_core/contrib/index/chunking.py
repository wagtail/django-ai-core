from typing import Protocol


class ChunkTransformer(Protocol):
    """Base class for chunking transformers which break a string into a list of strings."""

    def transform(self, text: "str") -> list["str"]:
        """Transform a string into chunks."""
        ...


class SimpleChunkTransformer(ChunkTransformer):
    """Simple character-based chunking transformer."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def transform(self, text: str) -> list[str]:
        chunks = []

        if len(text) <= self.chunk_size:
            chunks = [text]
            return chunks

        # Split into overlapping chunks
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]

            chunks.append(
                chunk_content,
            )

            # Move start position, accounting for overlap
            if end >= len(text):
                break
            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks


class SentenceChunkTransformer(ChunkTransformer):
    """Chunks strings using sentence-based splitting via LlamaIndex."""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def transform(self, text: str) -> list[str]:
        from llama_index.core import Document as LlamaDocument
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        llama_doc = LlamaDocument()
        llama_doc.set_content(text)
        nodes = splitter.get_nodes_from_documents([llama_doc])

        chunks = []
        for _, node in enumerate(nodes):
            chunks.append(
                node.get_content(),
            )
        return chunks


class ParagraphChunkTransformer(ChunkTransformer):
    """Chunks strings by paragraphs, combining small ones."""

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def transform(self, text: str) -> list[str]:
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finalize current chunk
            if (
                current_chunk
                and len(current_chunk) + len(paragraph) > self.max_chunk_size
            ):
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    chunk_index += 1
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk if it exists
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())

        # If no chunks were created (all text too short), create one chunk anyway
        if not chunks and text.strip():
            chunks.append(
                text.strip(),
            )

        return chunks
