import json

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from testapp.models import Book, Film, VideoGame


class Command(BaseCommand):
    def import_obj(self, ref, model):
        with open(f"tests/testapp/fixtures/{ref}_data.json") as f:
            data = json.load(f)
            for obj in data:
                model.objects.create(title=obj["title"], description=obj["description"])

    def create_superuser(self):
        User.objects.create_superuser(
            username="admin", email="admin@example.com", password="admin"
        )

    def handle(self, *args, **options):
        self.import_obj("books", Book)
        self.import_obj("films", Film)
        self.import_obj("games", VideoGame)
        self.create_superuser()
