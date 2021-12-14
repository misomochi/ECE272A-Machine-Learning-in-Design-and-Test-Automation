import csv
import numpy as np

def load_data(filename = 'diabetes.csv'):
	with open(filename, 'r') as file:
		reader = csv.reader(file)
		columnNames = next(reader)
		rows = np.array(list(reader), dtype = float)

	return columnNames, rows

def separate_labels(columnNames, rows):
	if 'Outcome' in columnNames:
		labelColumnIndex = columnNames.index('Outcome')
		ys = rows[:, labelColumnIndex]
		xs = np.delete(rows, labelColumnIndex, axis = 1)
		del columnNames[labelColumnIndex]

		return columnNames, xs, ys
	else:
		return columnNames, rows, []

def save_prediction(filename, columnname, prediction):
	with open(filename, 'w', newline = '') as file:
		writer = csv.writer(file)
		writer.writerow(columnname)
		writer.writerows(map(lambda x: [x], prediction))

def save_scatterplot(filename, columnname, prediction):
	with open(filename, 'w', newline = '') as file:
		writer = csv.writer(file)
		writer.writerow(columnname)
		writer.writerows(prediction)
