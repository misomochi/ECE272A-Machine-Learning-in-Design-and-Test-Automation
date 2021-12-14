import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.io import to_html
from sklearn.preprocessing import StandardScaler

def weighted_average(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a NBA dataframe and perform a weighted average
    on player stats across multiple teams based on games played
    '''

    unique_player_df = pd.DataFrame(columns=df.columns)
    groups = df.groupby(by='Rk', axis=0)

    for i, group in groups:
        # player's Rk   Player  Pos Age Tm
        player_info = group.iloc[[0], :5]
        player_info = player_info.reset_index()

        # total games played and games started
        total_games_and_starts = group.iloc[:,5:7].sum().to_frame().transpose()

        # perform weighted average on player stats
        cumulative_stats = group.iloc[:, 7:].multiply(group['G'], axis='index').sum() 
        weighted_average_stats = (cumulative_stats / group['G'].sum()).to_frame().transpose()

        # new row containing weighted stats
        new_row = pd.concat(
            [player_info, total_games_and_starts, weighted_average_stats], axis=1)

        # append new row to new dataframe
        unique_player_df = unique_player_df.append(new_row)

    # clean up old indexes
    unique_player_df = unique_player_df.reset_index()
    unique_player_df = unique_player_df.drop(['level_0', 'index'], axis=1)

    return unique_player_df

def run(file_path, algo_path):
    df = pd.read_csv(file_path)
    df = df.fillna(0)

    unqiue_player_df = weighted_average(df)
    feature_extracted_df = pd.DataFrame(unqiue_player_df, columns = ['ORB', 'AST', 'BLK'])
    xs = feature_extracted_df.to_numpy()
    xs_scaled = StandardScaler().fit_transform(xs)

    ocsvm_clf = pickle.load(open(algo_path, 'rb'))
    scores = ocsvm_clf.score_samples(xs_scaled)
    prediction = ocsvm_clf.predict(xs_scaled)
    ### plot analysis ###
    result_df = pd.DataFrame(unqiue_player_df, columns = ['Player', 'ORB', 'AST', 'BLK'])
    result_df['Score'] = scores
    result_df['Prediction'] = prediction
    result_df.sort_values(by = ['Score'], inplace = True, ignore_index = True)
    result_df.loc[:2, 'Prediction'] = 0 # top 3 outlier
    result_df['Prediction'] = result_df['Prediction'].map({-1: 'Outlier', 0: 'Top 3 Outlier', 1: 'Inlier'})
    fig = px.scatter_3d(result_df, x = 'ORB', y = 'AST', z = 'BLK', color = 'Prediction')
    #fig.show()

    return to_html(fig, full_html = False)