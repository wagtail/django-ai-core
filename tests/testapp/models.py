from django.db import models


class Book(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()


class Film(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()


class VideoGame(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()
