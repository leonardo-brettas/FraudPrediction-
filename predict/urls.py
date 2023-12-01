from django.urls import path

from .api import router as api

urlpatterns = [
    path('', api.urls),
]