import pandas as pd
import numpy as np
from feature_engineering.add_elo_features import add_elo_features
from feature_engineering.add_head_to_head_features import add_head_to_head_features
from feature_engineering.add_player_performance_features import add_player_performance_features
from feature_engineering.reshape_to_player_vs_player import reshape_to_player_vs_player
from feature_engineering.add_feature_differences import add_feature_differences
from feature_engineering.encode_one_hot import encode_one_hot
from feature_engineering.encode_indoor import encode_indoor
from config import CLEANED_DATA_PATH, TRAIN_FILE, TEST_FILE

def feature_pipeline(df):
    
    # Create match_id as unique identifier for each match
    df["match_id"] = range(len(df))
    df.insert(0, "match_id", df.pop("match_id"))

    # Add features to the dataset
    df = add_elo_features(df)
    df = add_head_to_head_features(df)
    df = add_player_performance_features(df)
    
    # Reshape the dataset to player vs player format and add feature differences
    df = reshape_to_player_vs_player(df)
    df = add_feature_differences(df)
    df = encode_indoor(df)
    df = encode_one_hot(df)
    return df


def build_train_test_sets(df, test_size):

    # Split the dataset into training and testing sets based on the specified test size
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]   
    test = df.iloc[split_index:]

    # Drop unnecessary and leakages columns from the datasets
    drop_cols = ['match_id', 'tourney_id', 'tourney_name', 'draw_size',
       'tourney_date', 'match_num', 'score', 'best_of', 'round', 'minutes',
       'p1_id', 'p2_id', 'p1_name', 'p2_name', 'p1_hand', 'p2_hand', 'p1_ht',
       'p2_ht', 'p1_ioc', 'p2_ioc', 'p1_age', 'p2_age',
       'p1_rank_points', 'p2_rank_points', 'p1_ace', 'p2_ace', 'p1_df',
       'p2_df', 'p1_svpt', 'p2_svpt', 'p1_1stIn', 'p2_1stIn', 'p1_1stWon',
       'p2_1stWon', 'p1_2ndWon', 'p2_2ndWon', 'p1_SvGms', 'p2_SvGms',
       'p1_bpSaved', 'p2_bpSaved', 'p1_bpFaced', 'p2_bpFaced', 'p1_elo_surface_after','p2_elo_surface_after','p1_elo_after','p2_elo_after']

    train = train.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)
    return train, test

# Load the cleaned dataset and create training and testing sets
df = pd.read_csv(CLEANED_DATA_PATH)
df = feature_pipeline(df)
train, test = build_train_test_sets(df, test_size=0.2)
train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)
