import os
from datetime import date
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework import status

from .models import FileModel, AlgorithmModel, AnalyticModel
from .serializers import FileSerializer, AnalyticSerializer

# Create your views here.
def get_file_list():
	files = FileModel.objects.all()
	return files.distinct()

def get_algorithm_list():
	algorithms = AlgorithmModel.objects.all()
	return algorithms.distinct()

def get_algorithm(name):
	return AlgorithmModel.objects.get(algorithm_name = name)

def get_analytic_list():
	analytics = AnalyticModel.objects.all()
	return analytics.distinct()

def run_analytic(file_path, algo_object):
	algo_model = algo_object.saved_model
	algo_script = algo_object.inference_script
	file_abs_path = os.path.join(settings.MEDIA_ROOT, str(file_path))
	algo_model_path = os.path.join(settings.MEDIA_ROOT, str(algo_model))

	algo_model_import_path = str(algo_script).replace('/', '.').replace('//', '.')[:-3]
	algo_model_import_path = 'media.' + algo_model_import_path
	print('Running algorithm from: ', algo_model_import_path)

	algo_model = __import__(algo_model_import_path, fromlist = ['run'])

	return algo_model.run(file_abs_path, algo_model_path)

class MainView(APIView):
	parser_classes = [JSONParser, FormParser, MultiPartParser]
	renderer_classes = [TemplateHTMLRenderer]
	template_name = 'index.html'

	# See Django REST Request class here:
	# https://www.django-rest-framework.org/api-guide/requests/

	def get(self, request):
		return Response({'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list()}, status=status.HTTP_200_OK)

	def post(self, request):
		# Upload form
		if 'upload' in request.data:
			# check for duplicated file names
			files = get_file_list()
			for file in files:
				if file.file_name == request.data['file_name']:
					if os.path.isfile(file.file_content.path):
						os.remove(file.file_content.path)
					file.delete()
					break

			file_serializer = FileSerializer(data=request.data)

			if file_serializer.is_valid():
				file_serializer.save()
				return Response({'status': 'Upload successful!', 'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list()}, status=status.HTTP_201_CREATED)
			else:
				# Validate file extension (.csv) before uploading in serializers.py
				return Response({'status': 'File is not CSV.', 'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list()}, status=status.HTTP_400_BAD_REQUEST)
		elif 'delete' in request.data:
			file = FileModel.objects.get(pk = request.data['delete'])

			if os.path.isfile(file.file_content.path): 
				os.remove(file.file_content.path)
				file.delete()
			else:
				return Response({'status': 'File deletion failed.', 'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list()}, status=status.HTTP_400_BAD_REQUEST)

			return Response({'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list()}, status = status.HTTP_200_OK)
		elif 'analytic' in request.data:
			# Run analytics on dataset as specified by file_name and analytic_id received in the post request
			query_file_name = request.data['file_name']
			query_algorithm = request.data['algorithm']

			# Find file path to local folder
			file_obj = FileModel.objects.get(file_name = query_file_name)
			file_path = file_obj.file_content
			print('File path: ', file_path)

			# Find algorithm
			algo_obj = get_algorithm(query_algorithm)
			analytic_result = run_analytic(file_path, algo_obj) # return graph html
			
			# save as history list
			to_save = {'analytic_name': query_file_name + '_' + query_algorithm, 'dataset_name': query_file_name, 'algorithm_name': query_algorithm, 'plot_html': analytic_result}
			analytic_serializer = AnalyticSerializer(data = to_save)

			if analytic_serializer.is_valid():
				analytic_serializer.save()

				return Response({'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list(), 'result_plot': analytic_result}, status = status.HTTP_200_OK)
			else:
				# HttpResponse is imported from django.http
				return HttpResponse('The server encountered an internal error while processing ' + to_save['analytic_name'], status=status.HTTP_500_INTERNAL_SERVER_ERROR)
		elif 'get_plot' in request.data:
			analytic_obj = AnalyticModel.objects.get(pk = request.data['get_plot'])

			return Response({'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list(), 'result_plot': analytic_obj.plot_html}, status = status.HTTP_200_OK)
		elif 'delete_entry' in request.data:
			entry = AnalyticModel.objects.get(pk = request.data['delete_entry'])
			entry.delete()

			return Response({'files': get_file_list(), 'algorithms': get_algorithm_list(), 'analytic_history': get_analytic_list()}, status = status.HTTP_200_OK)
		else:
			return Response(status = status.HTTP_400_BAD_REQUEST)