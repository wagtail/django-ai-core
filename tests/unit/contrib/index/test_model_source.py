import random

import pytest
from testapp.models import Book

from django_ai_core.contrib.index.models import ModelSourceIndex
from django_ai_core.contrib.index.source import ModelSource


def generate_long_string(length=2000):
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "data",
        "science",
        "python",
        "django",
        "wagtail",
        "testing",
        "development",
        "software",
        "application",
        "framework",
        "database",
        "model",
        "query",
        "index",
        "search",
        "semantic",
        "vector",
        "embedding",
        "algorithm",
        "neural",
        "network",
        "training",
        "optimization",
        "performance",
        "scalability",
    ]
    text = []
    current_length = 0
    while current_length < length:
        word = random.choice(words)
        text.append(word)
        current_length += len(word) + 1
    return " ".join(text).capitalize() + "."


@pytest.mark.django_db
def test_model_source_returns_unique_keys():
    for _ in range(5):
        Book.objects.create(title="Book Title", description=generate_long_string())

    model_source = ModelSource(model=Book)
    documents = model_source.get_documents()
    document_keys = [doc.document_key for doc in documents]
    assert len(document_keys) == len(set(document_keys))


@pytest.mark.django_db
def test_model_source_registers_index_by_class_name():
    class VerboseIndex:
        def __repr__(self):
            return "verbose index representation" * 100

    Book.objects.create(title="Book Title", description="Description")

    source = ModelSource(model=Book)
    source.post_index_update(VerboseIndex())
    source.post_index_update(VerboseIndex())

    registration = ModelSourceIndex.objects.get()
    assert ModelSourceIndex.objects.count() == 1
    assert registration.index_name == "VerboseIndex"
