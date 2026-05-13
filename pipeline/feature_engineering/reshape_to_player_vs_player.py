import pandas as pd
import numpy as np

def reshape_to_player_vs_player(df):
    #Create target variable
    #Input: dataframe with columns for winner and loser: "w_id", "w_name", ..., "l_id", "l_name", ...
    # -> Convert into format "p1_id", "p2_id", "p1_name", "p2_name", ... and randomly assign winner/loser to p1/p2 for each row

    #df = df.copy()

    #Create columns p1_... and p2_...
    swap = np.random.rand(len(df)) < 0.5 #for each row: True: p1 = winner, p2 = loser / False: p1 = loser, p2 = winner
    for col in df.columns:
        if col.startswith("w_"):
            feature = col.replace("w_", "")
            loser_col = f"l_{feature}"
            if loser_col in df.columns:
                df[f"p1_{feature}"] = np.where(swap, df[col], df[loser_col])
                df[f"p2_{feature}"] = np.where(swap, df[loser_col], df[col])

    #Create target variable Winner_is_p1: 1 if player 1 won, 0 if player 2 won
    df["Winner_is_p1"] = swap.astype(int)

    # Drop original w_ and l_ columns
    drop_cols = [col for col in df.columns if col.startswith("w_") or col.startswith("l_")]
    df = df.drop(columns=drop_cols)
    return df