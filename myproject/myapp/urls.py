from django.urls import path
from .views import home, predict_video  
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_video, name='predict_video'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)