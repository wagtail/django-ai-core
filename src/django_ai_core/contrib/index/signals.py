from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import ModelSourceIndex
from .source import ObjectSource


@receiver(post_save)
def handle_model_save(sender, instance, **kwargs):
    """When a model is saved, update relevant indexes."""

    # Get indexes this instance is currently registered in
    registered_indexes = ModelSourceIndex.get_indexes_for_object(instance)

    for index in registered_indexes:
        for source in index.sources:
            if isinstance(source, ObjectSource) and source.provides_object(sender):
                documents = source.objects_to_documents(sender)
                index.update(documents)


@receiver(post_delete)
def handle_model_delete(sender, instance, **kwargs):
    """When a model is deleted, remove it from all indexes."""

    pass
    # registered_indexes = ModelSourceIndex.get_indexes_for_object(instance)

    # TODO: handle object deletion
