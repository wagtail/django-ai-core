from typing import ClassVar, Type, Literal
import uuid
from django.db import models

from pgvector.django import VectorField


class PgVectorModelMixin(models.Model):
    """
    Mixin for Django models to be used with PgVectorProvider.

    This mixin provides the structure needed for a Django model to be used as a vector
    storage backend with PgVectorProvider. It defines the necessary fields that must be
    included in the model.

    Example:
        class MyVectorModel(PgVectorModelMixin, models.Model):
            # Additional fields can be added here
            # You can also customize the embedding_field and dimension if needed

            class Meta:
                indexes = [
                    # Optionally add indexes for better performance
                    # See pgvector docs for available index types
                ]
    """

    # Fields required by PgVectorProvider
    document_key = models.CharField(max_length=255, primary_key=True)
    content = models.TextField()
    metadata = models.JSONField(default=dict)

    # The field name that contains the vector embedding
    embedding_field: ClassVar[str] = "embedding"
    # The dimensionality of the vector embedding (must be specified in subclasses)
    embedding_dimension: ClassVar[int] = 0

    class Meta:
        abstract = True

    @property
    def vector(self) -> list[float]:
        """
        Get the vector embedding as a list of floats.
        This property is used by the storage provider to access the embedding.
        """
        return getattr(self, self.embedding_field)

    @vector.setter
    def vector(self, value: list[float]) -> None:
        """
        Set the vector embedding.
        This property is used by the storage provider to set the embedding.
        """
        setattr(self, self.embedding_field, value)


def create_model(
    dimensions: int,
    app_label: str = "django_ai_core",
    model_name: str | None = None,
    index_type: Literal["hnsw", "ivfflat", None] = None,
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine",
    **index_params: dict[str, int | str],
) -> Type[PgVectorModelMixin]:
    """
    Dynamically create a Django model with PgVectorModelMixin and VectorField.

    This function creates a new Django model that can be used with PgVectorProvider.
    The model will have a VectorField with the specified dimensions and can include
    optional indexing configuration.

    Args:
        dimensions: The dimensionality of the vector embedding.
        app_label: The Django app label for the model.
        model_name: Optional custom name for the model. If not provided, a name will be generated.
        index_type: The type of index to create (hnsw, ivfflat, or None).
        distance_metric: The distance metric to use (cosine, l2, ip).
        **index_params: Additional parameters for the index configuration.
            - For HNSW: m (int), ef_construction (int)
            - For IVFFlat: lists (int)

    Returns:
        A Django model class with PgVectorModelMixin and VectorField.

    Example:
        ```python
        # Create a model with 768-dimensional vectors
        MyModel = create_model(dimensions=768)

        # Create a model with an HNSW index
        MyModel = create_model(
            dimensions=768,
            app_label="myapp",
            model_name="MyVectorModel",
            index_type="hnsw",
            m=16,
            ef_construction=64
        )

        # Use the model with PgVectorProvider
        storage = PgVectorProvider(model=MyModel)
        ```
    """
    # Generate a model name if not provided
    if model_name is None:
        model_name = f"PgVectorModel_{dimensions}d_{uuid.uuid4().hex[:8]}"

    # Define the Meta inner class for the model
    meta_attrs = {
        "app_label": app_label,
    }

    # Add index configuration if requested
    if index_type is not None:
        from pgvector.django import HnswIndex, IvfflatIndex

        # Determine opclass based on distance metric
        opclass = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }[distance_metric]

        # Create the index configuration
        if index_type == "hnsw":
            m = index_params.get("m", 16)
            ef_construction = index_params.get("ef_construction", 64)

            index = HnswIndex(
                name=f"{model_name.lower()}_hnsw_idx",
                fields=["embedding"],
                m=m,
                ef_construction=ef_construction,
                opclasses=[opclass],
            )
        elif index_type == "ivfflat":
            lists = index_params.get("lists", 100)

            index = IvfflatIndex(
                name=f"{model_name.lower()}_ivfflat_idx",
                fields=["embedding"],
                lists=lists,
                opclasses=[opclass],
            )
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        meta_attrs["indexes"] = [index]

    # Create the Meta class
    meta = type("Meta", (), meta_attrs)

    # Define model attributes
    attrs = {
        "Meta": meta,
        "embedding": VectorField(dimensions=dimensions),
        "embedding_field": "embedding",
        "embedding_dimension": dimensions,
        "__module__": f"{app_label}.models",
    }

    # Create and return the model class
    return type(model_name, (PgVectorModelMixin,), attrs)
