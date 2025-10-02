from typing import Self, Sequence

from django.db import models
from pgvector.django import CosineDistance, VectorField


class PgvectorEmbeddingQuerySet(models.QuerySet["BasePgVectorEmbedding"]):
    def annotate_with_distance(
        self,
        query_vector: Sequence[float],
    ) -> Self:
        kwargs = {"distance": CosineDistance("vector", query_vector)}
        return self.annotate(**kwargs)


class PgvectorEmbeddingManager(models.Manager.from_queryset(PgvectorEmbeddingQuerySet)):
    pass


class BasePgVectorEmbedding(models.Model):
    """
    Django model to be used with PgVectorProvider.
    """

    index_name = models.CharField(max_length=255)
    document_key = models.CharField(max_length=255, primary_key=True)
    content = models.TextField()
    metadata = models.JSONField(default=dict)

    objects = PgvectorEmbeddingManager()

    class Meta:
        abstract = True

    def __str__(self):
        return self.document_key


class PgVectorEmbedding(BasePgVectorEmbedding):
    vector = VectorField()
