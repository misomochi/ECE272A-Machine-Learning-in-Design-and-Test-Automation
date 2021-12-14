import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor, kernels

columnNames, data = preprocessing.load_data("white_wine.csv")
columnNames, xs, ys = preprocessing.separate_labels(columnNames, data)
xs_scaled = StandardScaler().fit_transform(xs)
xs_new = SelectKBest(k = 9).fit_transform(xs_scaled, ys) # citric acid, sulphate deleted

# plot histogram of quality variable
plt.hist(ys, density = False, bins = 6)
plt.title('Distribution of Quality Variable')
plt.ylabel('Count')
plt.xlabel('Quality')
plt.show()

# plot feature correlation matrix
df = pd.DataFrame(data, columns = columnNames)
correlation_mat = df.corr()
sns.heatmap(correlation_mat, annot = True)
plt.show()

xs_train, xs_validation, ys_train, ys_validation = train_test_split(xs_new, ys, random_state = 0, stratify = ys)

# Logistic Regression
"""
# hyperparameters tuning
lr_param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2', 'elasticnet'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
lr_grid_search = GridSearchCV(estimator = LogisticRegression(multi_class = 'multinomial', n_jobs = -1), param_grid = lr_param_grid, n_jobs = -1, cv = 11, return_train_score = True)
lr_grid_search.fit(xs_train, ys_train)
print(lr_grid_search.best_params_)
print(lr_grid_search.score(xs_validation, ys_validation))
"""
lr_clf = LogisticRegression(C = 1, penalty = 'l1', solver = 'saga')
lr_result = lr_clf.fit(xs_train, ys_train)
print("LR accuracy is %0.4f" % lr_result.score(xs_validation, ys_validation)) # 0.5318

# Linear Regression
"""
# hyperparameters tuning
tr_param_grid = {'power': [0, 1, 2, 3]}
tr_grid_search = GridSearchCV(estimator = TweedieRegressor(), param_grid = tr_param_grid, n_jobs = -1, cv = 11, return_train_score = True)
tr_grid_search.fit(xs_train, ys_train)
print(tr_grid_search.best_params_)
print(tr_grid_search.score(xs_validation, ys_validation))
"""
tr_clf = TweedieRegressor(power = 1)
tr_result = tr_clf.fit(xs_train, ys_train)
print("TR accuracy is %0.4f" % tr_result.score(xs_validation, ys_validation)) # 0.2761

# Multi Layer Perceptrons Classifier
"""
# hyperparameters tuning
mlpc_param_grid = {'hidden_layer_sizes': [(10, 9, 8, 7, 6, 5), (10, 9, 8, 7, 6)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'max_iter': range(450, 550, 10)}
mlpc_grid_search = GridSearchCV(estimator = MLPClassifier(random_state = 0, early_stopping = True), param_grid = mlpc_param_grid, n_jobs = -1, cv = 5, return_train_score = True)
mlpc_grid_search.fit(xs_train, ys_train)
print(mlpc_grid_search.best_params_)
print(mlpc_grid_search.score(xs_validation, ys_validation))
"""
mlpc_clf = MLPClassifier(hidden_layer_sizes = (100, 100, 100, 100, 100), activation = 'relu', solver = 'adam', random_state = 0, alpha = 0.0001, batch_size = 128)
mlpc_result = mlpc_clf.fit(xs_train, ys_train)
print("MLPC accuracy is %0.4f" % mlpc_result.score(xs_validation, ys_validation)) # 0.6055

# Multi Layer Perceptrons Regressor
"""
# hyperparameters tuning
mlpr_param_grid = {'hidden_layer_sizes': [(10, 9, 8, 7, 6, 5), (10, 9, 8, 7, 6), (10, 9, 8, 7)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate': ['constant', 'invscaling', 'adaptive']}
mlpr_grid_search = GridSearchCV(estimator = MLPRegressor(), param_grid = mlpr_param_grid, n_jobs = -1, cv = 5, return_train_score = True)
mlpr_grid_search.fit(xs_train, ys_train)
print(mlpr_grid_search.best_params_)
print(mlpr_grid_search.score(xs_validation, ys_validation))
"""
mlpr_clf = MLPRegressor(hidden_layer_sizes = (100, 100, 100, 100, 100), activation = 'relu', solver = 'adam', random_state = 0, early_stopping = True)
mlpr_result = mlpr_clf.fit(xs_train, ys_train)
print("MLPR accuracy is %0.4f" % mlpr_result.score(xs_validation, ys_validation)) # 0.3753

# Support Vector Classification
"""
# hyperparameters tuning
svc_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'], 'decision_function_shape': ['ovo', 'ovr']}
svc_grid_search = GridSearchCV(estimator = SVC(random_state= 0), param_grid = svc_param_grid, n_jobs = -1, cv = 5, return_train_score = True)
svc_grid_search.fit(xs_train, ys_train)
print(svc_grid_search.best_params_)
print(svc_grid_search.score(xs_validation, ys_validation))
"""
svc_clf = SVC(C = 10, kernel = 'rbf', gamma = 'scale', decision_function_shape = 'ovo', random_state = 0)
svc_result = svc_clf.fit(xs_train, ys_train)
print("SVC accuracy is %0.4f" % svc_result.score(xs_validation, ys_validation)) # 0.5830

# Support Vector Regression
"""
# hyperparameters tuning
svr_param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['scale', 'auto'], 'C': [0.1, 1, 10]}
svr_grid_search = GridSearchCV(estimator = SVR(), param_grid = svr_param_grid, n_jobs = -1, cv = 5, return_train_score = True)
svr_grid_search.fit(xs_train, ys_train)
print(svr_grid_search.best_params_)
print(svr_grid_search.score(xs_validation, ys_validation))
"""
svr_clf = SVR(kernel = 'rbf', gamma = 'auto')
svr_result = svr_clf.fit(xs_train, ys_train)
print("SVR accuracy is %0.4f" % svr_result.score(xs_validation, ys_validation)) # 0.4031

# Gaussian Process Classifier
"""
# hyperparameters tuning
gpc_param_grid = {'multi_class': ['one_vs_rest', 'one_vs_one']}
# 'kernel': [1*kernels.RBF(), 1*kernels.Matern(), 1*kernels.RationalQuadratic(), 1*kernels.DotProduct()]
gpc_grid_search = GridSearchCV(estimator = GaussianProcessClassifier(kernel = 1*kernels.Matern(), random_state = 0, n_jobs = -1), param_grid = gpc_param_grid, n_jobs = -1, cv = 5, return_train_score = True)
gpc_grid_search.fit(xs_train, ys_train)
print(gpc_grid_search.best_params_)
print(gpc_grid_search.score(xs_validation, ys_validation))
"""
gpc_clf = GaussianProcessClassifier(kernel = kernels.RationalQuadratic() * kernels.DotProduct(), random_state = 0, multi_class = 'one_vs_rest', n_jobs = -1)
gpc_result = gpc_clf.fit(xs_train, ys_train)
print("GPC accuracy is %0.4f" % gpc_result.score(xs_validation, ys_validation))

# Gaussian Process Regressor
gpr_clf = GaussianProcessRegressor(random_state = 0)
gpr_result = gpr_clf.fit(xs_train, ys_train)
print("GPR accuracy is %0.4f" % gpr_result.score(xs_validation, ys_validation)) # -0.7653
"""
# cross validation
clf = [lr_clf, tr_clf, mlpc_clf, mlpr_clf, svc_clf, svr_clf, gpr_clf]
clf_name = ["lr_clf", "tr_clf", "mlpc_clf", "mlpr_clf", "svc_clf", "svr_clf", "gpr_clf"]

scores = cross_val_score(gpc_clf, xs_train, ys_train, cv = 5, n_jobs = -1)
print(scores)
print("GPC has %0.4f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

for i in range(10, 55, 5):
	ss = ShuffleSplit(n_splits=11, test_size=i / 100, random_state=0)
	print("test_size = %0.2f" % (i/100))
	for j in range(7):
		scores = cross_val_score(clf[j], xs_train, ys_train, cv = ss, n_jobs = -1)
		print(scores)
		print("%s has %0.4f accuracy with a standard deviation of %0.2f" % (clf_name[j], scores.mean(), scores.std()))
	print("---------------------------------------------------------")
"""

### Prediction ###
columnNames, data = preprocessing.load_data("unknowns.csv")
columnNames, xs_test, ys_test = preprocessing.separate_labels(columnNames, data)
df = pd.DataFrame(xs_test, columns = columnNames)
df.drop(['citric acid', 'sulphates'], axis = 1, inplace = True)
xs_test_new = df.to_numpy()
xs_test_scaled = StandardScaler().fit_transform(xs_test_new)
gpc_clf = GaussianProcessClassifier(kernel = kernels.RationalQuadratic() * kernels.DotProduct(), random_state = 0, multi_class = 'one_vs_rest', n_jobs = -1)
gpc_result = gpc_clf.fit(xs_new, ys)
prediction = gpc_result.predict(xs_test_scaled)
preprocessing.save_prediction("scores.csv", '', prediction)