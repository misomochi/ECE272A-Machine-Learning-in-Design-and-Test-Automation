import pandas as pd
import numpy as np


def weighted_average(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Takes a NBA dataframe and perform a weighted average
    on player stats across multiple teams based on games played
    '''

    unique_player_df = pd.DataFrame(columns=df.columns)
    groups = df.groupby(by='Rk', axis=0)

    for i, group in groups:
        # player's Rk	Player	Pos	Age	Tm
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