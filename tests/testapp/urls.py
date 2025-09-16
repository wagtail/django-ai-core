from django.contrib import admin
from django.urls import path, include
from django_ai_core.contrib.agents.urls import agent_urls as ai_core_agent_urls

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("ai/", include(ai_core_agent_urls())),
]
