from django.db import models
from pgvector.django import VectorField


class BasePgVectorEmbedding(models.Model):
    """
    Django model to be used with PgVectorProvider.
    """

    index_name = models.CharField(max_length=255)
    document_key = models.CharField(max_length=255, primary_key=True)
    content = models.TextField()
    metadata = models.JSONField(default=dict)

    class Meta:
        abstract = True

    def __str__(self):
        return self.document_key


class PgVectorEmbedding(BasePgVectorEmbedding):
    vector = VectorField()
