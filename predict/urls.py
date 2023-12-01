from django.urls import path

from .api import api as api

urlpatterns = [
    path('', api.urls),
]