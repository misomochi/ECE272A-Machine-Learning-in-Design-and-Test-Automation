import csv
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import to_html
from sklearn.preprocessing import StandardScaler

def load_data(filename):
	with open(filename, 'r') as file:
		reader = csv.reader(file)
		columnNames = next(reader)
		rows = np.array(list(reader), dtype = float)

	return columnNames, rows

def separate_labels(columnNames, rows, outcome):
	if outcome in columnNames:
		labelColumnIndex = columnNames.index(outcome)
		ys = rows[:, labelColumnIndex]
		xs = np.delete(rows, labelColumnIndex, axis = 1)
		del columnNames[labelColumnIndex]

		return columnNames, xs, ys
	else:
		return columnNames, rows, []

def run(file_path, algo_path):
	columnNames, data = load_data(file_path) # unknowns.csv
	columnNames, xs_test, ys_test = separate_labels(columnNames, data, 'quality')

	df = pd.DataFrame(xs_test, columns = columnNames)
	df.drop(['citric acid', 'sulphates'], axis = 1, inplace = True)
	xs_test_new = df.to_numpy()
	xs_test_scaled = StandardScaler().fit_transform(xs_test_new)

	gcp_clf = pickle.load(open(algo_path, 'rb'))
	prediction = gcp_clf.predict(xs_test_scaled)
	
	### plot analysis ###
	result_df = pd.DataFrame(ys_test, columns = ['quality'])
	result_df['prediction'] = prediction
	fig = px.scatter(result_df, x = 'quality', y = 'prediction', opacity = 0.2)
	#fig.show()

	return to_html(fig, full_html = False)