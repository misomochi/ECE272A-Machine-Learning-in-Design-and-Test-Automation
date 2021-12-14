import csv
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import to_html
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
	columnNames, xs_test, ys_test = separate_labels(columnNames, data, 'Outcome')

	imp = IterativeImputer(missing_values = 0, max_iter=10, random_state=0)
	xs_test = imp.fit_transform(xs_test)

	lda_clf = pickle.load(open(algo_path, 'rb'))
	prediction = lda_clf.predict(xs_test)

	### plot analysis ###
	result_df = pd.DataFrame(xs_test, columns = columnNames)
	result_df['Prediction'] = prediction
	result_df['Prediction'] = result_df['Prediction'].map({0.0: 'Negative', 1.0: 'Positive'})
	fig = px.scatter(result_df, x = 'Glucose', y = 'BloodPressure', color = 'Prediction')
	#fig.show()
	
	return to_html(fig, full_html = False)