# Chunk Transformers

Chunk transformers are utilities for breaking strings in to smaller strings.

```python
chunk_transformer = SimpleChunkTransformer(chunk_size=1000, chunk_overlap=100)

big_string = """
lots of data....
"""

chunks = chunk_transformer.transform(big_string)
```

## SimpleChunkTransformer

Breaks text in to fixed chunks of size `chunk_size` characters, where each chunk overlaps by `chunk_overlap` characters.

## SentenceChunkTransformer

Breaks text in to fixed chunks, attempting to keep sentence boundaries.

## ParagraphChunkTransformer

Chunks text by paragraph, combining smaller paragraphs.

## Custom Transformers

A chunk transformer only needs to implement one method: `transform` which takes a string and returns a list of strings.

```python

class MyChunkTransformer:
    def transform(self, text):
        return text.split(":")
```
