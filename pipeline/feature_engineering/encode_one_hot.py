import pandas as pd

def encode_one_hot(df):
    # One-hot encode the 'surface' column
    df = df.copy()
    df = pd.get_dummies(df, columns=["surface"])
    df = pd.get_dummies(df, columns=["tourney_level"])
    return df