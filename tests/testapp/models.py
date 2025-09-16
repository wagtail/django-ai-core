from django.db import models

from src.django_ai_core.contrib.index.storage.pgvector_model import create_model


class Book(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()


class Film(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()


class VideoGame(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()


MediaVectorModel = create_model(
    dimensions=1536,
    app_label="testapp",
    model_name="MediaVectorModel",
    index_type="hnsw",
)
