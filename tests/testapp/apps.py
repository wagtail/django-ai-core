from django.apps import AppConfig


class DjangoAiTestAppConfig(AppConfig):
    label = "testapp"
    name = "testapp"
    verbose_name = "Django AI tests"

    def ready(self):
        from . import indexes  # noqa
        from . import agents  # noqa
