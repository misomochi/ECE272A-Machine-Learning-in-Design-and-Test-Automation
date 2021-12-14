# https://www.django-rest-framework.org/api-guide/serializers/

from .models import FileModel, AnalyticModel
from rest_framework import serializers

class FileSerializer(serializers.ModelSerializer):
	class Meta:
		model = FileModel
		fields = '__all__'

class AnalyticSerializer(serializers.ModelSerializer):
	class Meta:
		model = AnalyticModel
		fields = '__all__'