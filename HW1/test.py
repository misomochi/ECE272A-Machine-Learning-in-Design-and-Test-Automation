import preprocessing
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
#from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


columnNames, data = preprocessing.load_data("diabetes.csv")
columnNames, xs, ys = preprocessing.separate_labels(columnNames, data)

xs_train, xs_validation, ys_train, ys_validation = train_test_split(xs, ys, random_state = 0, stratify = ys)

# fill missing values
#imp = SimpleImputer(missing_values = 0, strategy = 'median') # 'mean', 'most_frequent', 'median'
imp = IterativeImputer(missing_values = 0, max_iter=10, random_state=0)
xs_train = imp.fit_transform(xs_train)

# feature transformation/normalization/selection

# Decision Tree
dt_clf = DecisionTreeClassifier()
dt_result = dt_clf.fit(xs_train, ys_train)
print("DT accuracy is %0.4f" % dt_result.score(xs_validation, ys_validation))
# plot decision tree
plot_tree(dt_result, feature_names = columnNames, filled = True, fontsize = 5)
plt.savefig('DTimage.png', dpi = 1200, bbox_inches = 'tight')

# create confusion matrix
cm = confusion_matrix(dt_clf.predict(xs_train), ys_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_clf.classes_)
disp.plot()
plt.savefig('DT_train_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')
cm = confusion_matrix(dt_clf.predict(xs_validation), ys_validation)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt_clf.classes_)
disp.plot()
plt.savefig('DT_validation_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')

# Linear Discriminant Analysis
lda_clf = LinearDiscriminantAnalysis()
lda_result = lda_clf.fit(xs_train, ys_train)
print("LDA accuracy is %0.4f" % lda_result.score(xs_validation, ys_validation))

# create confusion matrix
cm = confusion_matrix(lda_clf.predict(xs_train), ys_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda_clf.classes_)
disp.plot()
plt.savefig('LDA_train_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')
cm = confusion_matrix(lda_clf.predict(xs_validation), ys_validation)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lda_clf.classes_)
disp.plot()
plt.savefig('LDA_validation_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')


# Naive Bayes
gnb_clf = GaussianNB()
gnb_result = gnb_clf.fit(xs_train, ys_train)
print("CNB accuracy is %0.4f" % gnb_result.score(xs_validation, ys_validation))

# create confusion matrix
cm = confusion_matrix(gnb_clf.predict(xs_train), ys_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb_clf.classes_)
disp.plot()
plt.savefig('GNB_train_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')
cm = confusion_matrix(gnb_clf.predict(xs_validation), ys_validation)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb_clf.classes_)
disp.plot()
plt.savefig('GNB_validation_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')

# K Nearest Neighbors
"""
# tune hyperparameters
knn_param_grid = {'n_neighbors': range(2, 11)}
knn_grid_search = GridSearchCV(estimator = KNeighborsClassifier(weights = 'uniform', p = 1), param_grid = knn_param_grid, n_jobs = -1, cv = 5, verbose = 4, return_train_score = True)
knn_grid_search.fit(xs_train, ys_train)
print(knn_grid_search.best_params_)
print(knn_grid_search.score(xs_validation, ys_validation))
"""
knn_clf = KNeighborsClassifier(n_neighbors = 7, weights = 'uniform', p = 1)
knn_result = knn_clf.fit(xs_train, ys_train)
print("KNN accuracy is %0.4f" % knn_result.score(xs_validation, ys_validation))

# create confusion matrix
cm = confusion_matrix(knn_clf.predict(xs_train), ys_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_clf.classes_)
disp.plot()
plt.savefig('KNN_train_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')
cm = confusion_matrix(knn_clf.predict(xs_validation), ys_validation)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_clf.classes_)
disp.plot()
plt.savefig('KNN_validation_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')


# Support Vector Machine
"""
# tune hyperparameters
svm_param_grid = {'degree': range(1,5)}
svm_grid_search = GridSearchCV(estimator = SVC(C = 10, kernel = 'poly'), param_grid = svm_param_grid, n_jobs = -1, cv = 5, verbose = 4, return_train_score = True)
svm_grid_search.fit(xs_train, ys_train)
print(svm_grid_search.best_params_)
print(svm_grid_search.score(xs_validation, ys_validation))
"""
svm_clf = SVC(C = 10, kernel = 'poly', degree = 2)
svm_result = svm_clf.fit(xs_train, ys_train)
print("SVM accuracy is %0.4f" % svm_result.score(xs_validation, ys_validation))

# create confusion matrix
cm = confusion_matrix(svm_clf.predict(xs_train), ys_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_clf.classes_)
disp.plot()
plt.savefig('SVM_train_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')
cm = confusion_matrix(svm_clf.predict(xs_validation), ys_validation)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_clf.classes_)
disp.plot()
plt.savefig('SVM_validation_confusion_matrix.png', dpi = 1200, bbox_inches = 'tight')
"""
# cross validation
for i in range(10, 55, 5):
	ss = ShuffleSplit(n_splits=10, test_size=i / 100, random_state=0)
	print("test_size = %0.2f" % (i/100))
	scores = cross_val_score(dt_clf, xs_train, ys_train, cv = ss)
	print("DT has %0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
	scores = cross_val_score(lda_clf, xs_train, ys_train, cv = ss)
	print("LDA has %0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
	scores = cross_val_score(gnb_clf, xs_train, ys_train, cv = ss)
	print("CNB has %0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
	scores = cross_val_score(knn_clf, xs_train, ys_train, cv = ss)
	print("KNN has %0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
	scores = cross_val_score(svm_clf, xs_train, ys_train, cv = ss)
	print("SVM has %0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
	print("---------------------------------------------------------")
"""

### Prediction ###
columnNames, data = preprocessing.load_data("unknowns.csv")
columnNames, xs_test, ys_test = preprocessing.separate_labels(columnNames, data)
# predict and save as csv
lda_final = lda_clf.fit(imp.fit_transform(xs), ys)
prediction = lda_result.predict(xs_test)
preprocessing.save_prediction("scores.csv", '', prediction)