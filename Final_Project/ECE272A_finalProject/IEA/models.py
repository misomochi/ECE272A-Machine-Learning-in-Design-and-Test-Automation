import os
from django.db import models
from django.core.validators import FileExtensionValidator

# Create your models here.
# https://docs.djangoproject.com/en/3.1/topics/db/models/

class FileModel(models.Model):
	file_name = models.CharField(max_length = 50)
	file_content = models.FileField(upload_to = "upload", validators = [FileExtensionValidator(allowed_extensions = ['csv'])]) # media/upload/file_name

class AlgorithmModel(models.Model):
	algorithm_name = models.CharField(max_length = 50) # algorithm name
	inference_script = models.FileField(upload_to = 'algorithms/scripts/') # inference script: code used to predict the data
	saved_model = models.FileField(upload_to = 'algorithms/saved_models/') # trained model: pickle dump

	"""docstring for AlgorithmModel"""
	def __str__(self):
		return self.algorithm_name

	def delete(self, *args, **kwargs):
		if os.path.isfile(self.inference_script.path):
			os.remove(self.inference_script.path)

		if os.path.isfile(self.saved_model.path):
			os.remove(self.saved_model.path)

		super().delete(*args, **kwargs)

class AnalyticModel(models.Model):
	analytic_name = models.CharField(max_length = 50)
	dataset_name = models.CharField(max_length = 50)
	algorithm_name = models.CharField(max_length = 50)
	date_and_time = models.DateTimeField(auto_now = True)
	plot_html = models.TextField(default = '')