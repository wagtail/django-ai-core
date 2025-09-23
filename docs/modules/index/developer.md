# Developer Docs

!!! note "Technical notes"

    This is intended as an overview of the structure of Vector Indexes and their components. You don't need to read this if you're implementing a Vector Index in your app, but you might be interested if you're looking to customise how your index works, or want to build your own versions of components.

    The APIs documented here are not 'public' and may change in future releases.

## Vector Index

Every Vector Index starts with a subclass of `VectorIndex`. This doesn't do much by itself, but acts as the container for the various components needed to make a Vector Index work:

-   Sources - Anything which produces `Document` objects to be ingested in to the index. This can be Django models, API responses, or anything else. Sources are also responsible for "chunking" source data.
-   Embedding Transformer - Turns `Document`s in to `EmbeddedDocuments`.
-   Storage Providers - Stores `Document`s in some vector store allows querying across that data.
-   Query Handler - Takes user queries, embeds them using an Embedding Transformer and then queries the underlying storage provider.

## Sources

A source provides the data to be ingested in to a Vector Index, breaking it up in to smaller `Document` objects.

A basic `Source` implements the following:

-   A `source_id` property for uniquely identifying this source
-   `get_documents` - returns all `Document`s for this Source. This might involve fetching something from somewhere, converting the results to plain text, chunking it and then returning multiple `Document`s.
-   `provides_document` - returns a boolean indicating whether the given `Document` is provided by this Source.

Many Sources translate some sort of object; a Django Model, a File object, a dictionary from a REST API in to `Documents`. To support this; an `ObjectSource` subclass is available which provides an extended interface:

-   `objects_to_documents` - Takes an iterable of source object and returns `Document` sfor them.
-   `objects_from_documents` - Takes an iterable of `Document` objects and returns an original object representation for them, likely using their key or some metadata on the `Document` to rebuild or re-fetch the original source.
-   `provides_object` - returns a boolean indicating whether this source object is provided by this Source.

## Embedding Transformers

Embedding Transformers handle adding embeddings to `Document`s (and plain strings where needed).

They implement:

-   A `transformer_id` property for uniquely identifying the transformer
-   `embed_string` for embedding a string
-   `embed_documents` for embedding multiple `Document` objects and returning `EmbeddedDocument`s

## Storage Providers

Storage providers implement an interface for storing documents in vector storage backends, as well as for querying from them.

The query interface uses Queryish so that results behave like Django QuerySets.

There's some auto-class generation oddities going on here so for clarity:

-   Each storage provider implements
    -   a subclass of `StorageProvider` (defining the provider configuration and how to add/delete data)
    -   a subclass of `BaseStorageQuerySet` (defining how to query data)
-   The `StorageProvider` class is provided with a `base_queryset_cls` classvar which tells it what base queryset to use.
-   The `StorageProvider`'s `document_cls` property generates a Queryish Virtual Model from `BaseStorageDocument`, setting a `base_queryset_class`
-   When a VirtualModel (`BaseStorageDocument` in this case) class is created, Queryish generates a new queryset class based on `base_queryset_class`.

A storage provider must implement:

A subclass of `StorageProvider` implementing:

-   `add` - Add `EmbeddedDocument`s to the storage
-   `delete` - Delete a list of document_keys from the storage
-   `clear` - Clear everything from the storage

A subclass of `BaseStorageQuerySet` implementing:

-   `run_query` as described by [Queryish](https://github.com/wagtail/queryish?tab=readme-ov-file#other-data-sources)
