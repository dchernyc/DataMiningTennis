import pandas as pd
import numpy as np

def add_feature_differences(df):
    # Compute differences between p1 and p2 features and add them as new features; difference is calculated as p1 - p2, so a positive value means that p1 has a higher value in this feature than p2
    df = df.copy()
    features = ["ht","age","rank", "rank_points","ace_rate_10", "df_rate_10", "bp_save_rate_10", "serve_win_rate_10", "1stWon_rate_10", "2ndWon_rate_10", "h2h", 'win_rate_10', 'elo_before', 'elo_surface_before', 'elo_surface_blended_before']

    for feature in features:
        p1_col = f"p1_{feature}"
        p2_col = f"p2_{feature}"
        
        if p1_col in df.columns and p2_col in df.columns:
            df[f"{feature}_diff"] = df[p1_col] - df[p2_col]
    return df