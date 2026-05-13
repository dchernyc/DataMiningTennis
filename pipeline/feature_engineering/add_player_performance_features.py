import pandas as pd
import numpy as np

def add_player_performance_features(df):
    df = df.copy()

    # Get only relevant columns and create separate dataframes for winners and losers, to compute features on player level
    df_winner = df[[ 'match_id', 'w_id', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn','w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_id']].copy()
    df_loser = df[[ 'match_id', 'l_id', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn','l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'w_id']].copy()

    df_winner["is_winner"] = 1
    df_loser["is_winner"] = 0

    # Rename columns of winner and loser dataframes to have the same column names for player level features
    df_winner = df_winner.rename(columns={
        # winner_
        "w_id": "player_id",
        'l_id' : 'opponent_id',
        # w_
        "w_ace": "ace",
        "w_df": "df",
        "w_svpt": "svpt",
        "w_1stIn": "1stIn",
        "w_1stWon": "1stWon",
        "w_2ndWon": "2ndWon",
        "w_SvGms": "SvGms",
        "w_bpSaved": "bpSaved",
        "w_bpFaced": "bpFaced"
    })
    df_loser = df_loser.rename(columns={
        # loser_
        "l_id": "player_id",
        'w_id' : 'opponent_id',
        # l_
        "l_ace": "ace",
        "l_df": "df",
        "l_svpt": "svpt",
        "l_1stIn": "1stIn",
        "l_1stWon": "1stWon",
        "l_2ndWon": "2ndWon",
        "l_SvGms": "SvGms",
        "l_bpSaved": "bpSaved",
        "l_bpFaced": "bpFaced"
    })

    # Combine winners and losers to calculate player-level features
    df_players = pd.concat([df_winner, df_loser])
    df_players = df_players.sort_values(['match_id'])

    # Compute match specific features for each player
    df_players["1stWon_rate"] = df_players["1stWon"] / df_players["1stIn"]
    df_players["2ndWon_rate"] = df_players["2ndWon"] / (df_players["svpt"]-df_players["1stIn"])
    df_players["serve_win_rate"] = (df_players["1stWon"] + df_players["2ndWon"]) / df_players["svpt"]
    df_players["ace_rate"] = df_players["ace"] / df_players["svpt"]
    df_players["df_rate"] = df_players["df"] / df_players["svpt"]
    df_players["bp_save_rate"]=df_players["bpSaved"] / df_players["bpFaced"]

    # Compute rolling averages for the last 10 matches for each player and each feature, shifted by 1 to avoid data leakage
    df_players[['1stWon_rate_10', '2ndWon_rate_10', 'ace_rate_10', 'df_rate_10', 'bp_save_rate_10', 'serve_win_rate_10', 'win_rate_10']] = df_players.groupby('player_id')[['1stWon_rate', '2ndWon_rate', 'ace_rate', 'df_rate', 'bp_save_rate', 'serve_win_rate', 'is_winner']].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()) 

    # Create data frames for winners and losers with the new features, then merge them back from player-level to match-level
    df_winner = df_players[df_players['is_winner'] == 1].drop(columns=['player_id', 'opponent_id', 'is_winner',"ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]).copy()
    df_loser = df_players[df_players['is_winner'] == 0].drop(columns=['player_id', 'opponent_id', 'is_winner',"ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]).copy()

    # Rename columns to indicate winner and loser features, then merge back to the original dataframe
    features = ["1stWon_rate_10", "2ndWon_rate_10", "ace_rate_10", "df_rate_10", "bp_save_rate_10", "serve_win_rate_10", "win_rate_10"]
    df_winner = df_winner.rename(columns={col: f"w_{col}" for col in features})
    df_loser  = df_loser.rename(columns={col: f"l_{col}" for col in features})
    w_cols = [f"w_{col}" for col in features]
    l_cols = [f"l_{col}" for col in features]
    df_features = df_winner[['match_id'] + w_cols].merge(df_loser[['match_id'] + l_cols], on="match_id")  
    df = df.merge(df_features, on="match_id", how="left")
    return df