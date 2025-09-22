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

from .models import Book, Film, VideoGame, MediaVectorModel


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
            yield from self.objects_to_documents(details)

    def get_document_key(self, obj):
        return f"pokemon:{obj['name']}"

    def objects_to_documents(self, objs):
        if isinstance(objs, dict):
            objs = [objs]

        for obj in objs:
            content = f"{obj['name']}\n"
            for entry in obj["flavor_text_entries"]:
                if entry["language"]["name"] == "en":
                    content += f"{entry['flavor_text']}\n"
            yield Document(
                document_key=self.get_document_key(obj),
                content=content,
                metadata={},
            )


@registry.register()
class MediaIndex(VectorIndex):
    sources = [
        ModelSource(
            model=Book,
            content_fields=["title", "description"],
            metadata_fields=["description"],
        ),
        ModelSource(model=Film),
        ModelSource(model=VideoGame),
    ]
    storage_provider = PgVectorProvider(model=MediaVectorModel)
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
    storage_provider = PgVectorProvider(model=MediaVectorModel)
    embedding_transformer = CachedEmbeddingTransformer(
        base_transformer=CoreEmbeddingTransformer(
            llm_service=LLMService(
                provider="openai",
                model="text-embedding-3-small",
            )
        ),
    )
