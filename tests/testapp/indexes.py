import requests
from django_ai_core.llm import LLMService
from django_ai_core.contrib.index import VectorIndex, registry
from django_ai_core.contrib.index import (
    CachedEmbeddingTransformer,
    CoreEmbeddingTransformer,
)
from django_ai_core.contrib.index.source import ModelSource, Source
from django_ai_core.contrib.index.schema import Document
from django_ai_core.contrib.index.storage import PgVectorProvider
from django_ai_core.contrib.index.storage.s3_vectors import S3VectorProvider

from .models import Book, Film, VideoGame


class PokemonSource(Source):
    def get_documents(self):
        base_url = "https://pokeapi.co/api/v2/"
        pokemon_list = requests.get(f"{base_url}pokemon-species/?limit=50").json()[
            "results"
        ]
        for pokemon in pokemon_list:
            details = requests.get(
                f"{base_url}/pokemon-species/{pokemon['name']}"
            ).json()
            content = f"{details['name']}\n"
            for entry in details["flavor_text_entries"]:
                if entry["language"]["name"] == "en":
                    content += f"{entry['flavor_text']}\n"
            key = f"pokemon:{details['name']}"
            yield Document(
                document_key=key,
                content=content,
                metadata={},
            )

    def provides_document(self, document: Document) -> bool:
        return document.document_key.startswith("pokemon")


@registry.register()
class MediaIndex(VectorIndex):
    sources = [
        ModelSource(
            model=Book,
            content_fields=["title", "description"],
            metadata_fields=["title", "description"],
        ),
        ModelSource(model=Film),
        ModelSource(model=VideoGame),
    ]
    storage_provider = S3VectorProvider(
        bucket_name="prototyping-vector-bucket", dimensions=1536
    )
    embedding_transformer = CachedEmbeddingTransformer(
        base_transformer=CoreEmbeddingTransformer(
            llm_service=LLMService(
                provider="openai",
                model="text-embedding-3-small",
            )
        ),
    )


@registry.register()
class PokemonIndex(VectorIndex):
    sources = [PokemonSource()]
    storage_provider = PgVectorProvider()
    embedding_transformer = CachedEmbeddingTransformer(
        base_transformer=CoreEmbeddingTransformer(
            llm_service=LLMService(
                provider="openai",
                model="text-embedding-3-small",
            )
        ),
    )
