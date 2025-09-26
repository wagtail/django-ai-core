import requests
from unittest.mock import Mock
from django.conf import settings
from any_llm import AnyLLM
from django_ai_core.llm import LLMService
from django_ai_core.contrib.index import VectorIndex, registry
from django_ai_core.contrib.index import (
    CachedEmbeddingTransformer,
    CoreEmbeddingTransformer,
)
from django_ai_core.contrib.index.source import ModelSource, Source
from django_ai_core.contrib.index.schema import Document

from .models import Book, Film, VideoGame


class MockAnyLLM(AnyLLM):
    def completion(self, *, model_id, messages):
        return "completion"

    def responses(self, *, model_id, input_data):
        return "responses"

    def _embedding(self, *, model_id, inputs):
        return [0, 1, 2]


storage_provider_setting = settings.AI_CORE_TESTAPP_STORAGE_PROVIDER
llm_provider_setting = settings.AI_CORE_TESTAPP_LLM_PROVIDER

if storage_provider_setting == "pgvector":
    from django_ai_core.contrib.index.storage.pgvector import PgVectorProvider

    storage_provider = PgVectorProvider()
elif storage_provider_setting == "s3vectors":
    from django_ai_core.contrib.index.storage.s3vectors import S3VectorProvider

    storage_provider = S3VectorProvider(
        bucket_name="prototyping-vector-bucket", dimensions=1536
    )
else:
    from django_ai_core.contrib.index.storage.inmemory import InMemoryProvider

    storage_provider = InMemoryProvider()

if llm_provider_setting == "openai":
    llm_embedding_service = LLMService.create(
        provider="openai", model="text-embedding-3-small"
    )
else:
    llm_embedding_service = LLMService(client=Mock(spec=AnyLLM), model="mock")


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
    storage_provider = storage_provider
    embedding_transformer = CachedEmbeddingTransformer(
        base_transformer=CoreEmbeddingTransformer(llm_service=llm_embedding_service),
    )


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
class PokemonIndex(VectorIndex):
    sources = [PokemonSource()]
    storage_provider = storage_provider
    embedding_transformer = CachedEmbeddingTransformer(
        base_transformer=CoreEmbeddingTransformer(llm_service=llm_embedding_service),
    )
