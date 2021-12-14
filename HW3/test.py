import weighted_average
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

# load csv and fill nan entries with 0
df = pd.read_csv('nba_players_stats_20_21_per_game.csv')
df = df.fillna(0)

unqiue_player_df = weighted_average.weighted_average(df)
print(unqiue_player_df)

# plot 3d plot
threedee = plt.figure().gca(projection='3d')
threedee.scatter(unqiue_player_df['ORB'], unqiue_player_df['AST'], unqiue_player_df['BLK'])
threedee.set_xlabel('ORB')
threedee.set_ylabel('AST')
threedee.set_zlabel('BLK')
plt.show()

feature_extracted_df = pd.DataFrame(unqiue_player_df, columns = ['ORB', 'AST', 'BLK']) # 3 important features

# plot historgram and cdf of features
fig1, axis1 = plt.subplots()
fig1.tight_layout()
feature_extracted_df.hist(ax=axis1, bins = 50, cumulative = False)
fig2, axis2 = plt.subplots()
fig2.tight_layout()
feature_extracted_df.hist(ax=axis2, bins = 50, cumulative = True, histtype = 'step', density = 1)
plt.show()

xs = feature_extracted_df.to_numpy()
xs_scaled = StandardScaler().fit_transform(xs)

### One Class SVM
ocsvm_clf = OneClassSVM(nu = 0.1)
ocsvm_clf.fit(xs_scaled)
ocsvm_result_df = pd.DataFrame(unqiue_player_df, columns = ['Player', 'ORB', 'AST', 'BLK'])
ocsvm_result_df['Scores'] = ocsvm_clf.score_samples(xs_scaled)
ocsvm_result_df['Predictions'] = ocsvm_clf.predict(xs_scaled)
ocsvm_result_df.sort_values(by = ['Scores'], inplace = True, ignore_index = True)
outlier_df = ocsvm_result_df[ocsvm_result_df['Predictions'] == -1]
inlier_df = ocsvm_result_df[ocsvm_result_df['Predictions'] == 1]
# generate 3d scatterplot
ocsvm_fig = plt.figure()
ocsvm_fig.tight_layout()
ocsvm_axis = ocsvm_fig.add_subplot(111, projection = '3d')
ocsvm_axis.scatter(inlier_df['ORB'], inlier_df['AST'], zs = inlier_df['BLK'], c = 'blue', label = 'Inliers')
ocsvm_axis.scatter(outlier_df.loc[3:, 'ORB'], outlier_df.loc[3:, 'AST'], zs = outlier_df.loc[3:, 'BLK'], c = 'green', label = 'Outliers')
ocsvm_axis.scatter(outlier_df.loc[:2, 'ORB'], outlier_df.loc[:2, 'AST'], zs = outlier_df.loc[:2, 'BLK'], c = 'red', label = 'Top 3 Outliers')
ocsvm_axis.set_xlabel('ORB')
ocsvm_axis.set_ylabel('AST')
ocsvm_axis.set_zlabel('BLK')
plt.title('One-Class SVM')
plt.legend()
plt.show()
# export to csv
ocsvm_result_df.drop(['ORB', 'AST', 'BLK', 'Predictions'], axis = 1, inplace = True)
ocsvm_result_df.to_csv("One_Class_SVM_Scores.csv", index = False)

### Elliptic Envelope
ee_clf = EllipticEnvelope(contamination = 0.1, random_state = 0)
ee_clf.fit(xs_scaled)
ee_result_df = pd.DataFrame(unqiue_player_df, columns = ['Player', 'ORB', 'AST', 'BLK'])
ee_result_df['Scores'] = ee_clf.score_samples(xs_scaled)
ee_result_df['Predictions'] = ee_clf.predict(xs_scaled)
ee_result_df.sort_values(by = ['Scores'], inplace = True, ignore_index = True)
outlier_df = ee_result_df[ee_result_df['Predictions'] == -1]
inlier_df = ee_result_df[ee_result_df['Predictions'] == 1]
# generate 3d scatterplot
ee_fig = plt.figure()
ee_fig.tight_layout()
ee_axis = ee_fig.add_subplot(111, projection = '3d')
ee_axis.scatter(inlier_df['ORB'], inlier_df['AST'], zs = inlier_df['BLK'], c = 'blue', label = 'Inliers')
ee_axis.scatter(outlier_df.loc[3:, 'ORB'], outlier_df.loc[3:, 'AST'], zs = outlier_df.loc[3:, 'BLK'], c = 'green', label = 'Outliers')
ee_axis.scatter(outlier_df.loc[:2, 'ORB'], outlier_df.loc[:2, 'AST'], zs = outlier_df.loc[:2, 'BLK'], c = 'red', label = 'Top 3 Outliers')
ee_axis.set_xlabel('ORB')
ee_axis.set_ylabel('AST')
ee_axis.set_zlabel('BLK')
plt.title('Elliptic Envelope')
plt.legend()
plt.show()
# export to csv
ee_result_df.drop(['ORB', 'AST', 'BLK', 'Predictions'], axis = 1, inplace = True)
ee_result_df.to_csv("Elliptical_Envelope_Scores.csv", index = False)

### Isolation Forest
if_clf = IsolationForest(contamination = 0.1, n_jobs = -1, random_state = 0)
if_clf.fit(xs_scaled)
if_result_df = pd.DataFrame(unqiue_player_df, columns = ['Player', 'ORB', 'AST', 'BLK'])
if_result_df['Scores'] = if_clf.score_samples(xs_scaled)
if_result_df['Predictions'] = if_clf.predict(xs_scaled)
if_result_df.sort_values(by = ['Scores'], inplace = True, ignore_index = True)
outlier_df = if_result_df[if_result_df['Predictions'] == -1]
inlier_df = if_result_df[if_result_df['Predictions'] == 1]
# generate 3d scatterplot
if_fig = plt.figure()
if_fig.tight_layout()
if_axis = if_fig.add_subplot(111, projection = '3d')
if_axis.scatter(inlier_df['ORB'], inlier_df['AST'], zs = inlier_df['BLK'], c = 'blue', label = 'Inliers')
if_axis.scatter(outlier_df.loc[3:, 'ORB'], outlier_df.loc[3:, 'AST'], zs = outlier_df.loc[3:, 'BLK'], c = 'green', label = 'Outliers')
if_axis.scatter(outlier_df.loc[:2, 'ORB'], outlier_df.loc[:2, 'AST'], zs = outlier_df.loc[:2, 'BLK'], c = 'red', label = 'Top 3 Outliers')
if_axis.set_xlabel('ORB')
if_axis.set_ylabel('AST')
if_axis.set_zlabel('BLK')
plt.title('Isolation Forest')
plt.legend()
plt.show()
# export to csv
if_result_df.drop(['ORB', 'AST', 'BLK', 'Predictions'], axis = 1, inplace = True)
if_result_df.to_csv("Isolation_Forest_Scores.csv", index = False)