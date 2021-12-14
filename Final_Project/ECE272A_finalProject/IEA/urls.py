#  https://docs.djangoproject.com/en/3.1/topics/http/urls/

from IEA import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = 'IEA'
urlpatterns = [
	path('', views.MainView.as_view(), name='IEA'), # localhost:8000/
]

if settings.DEBUG:
	urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)