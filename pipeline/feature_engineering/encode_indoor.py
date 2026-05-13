def encode_indoor(df):
    # Convert 'indoor' column to binary
    df = df.copy()
    df["indoor"] = df["indoor"].map({"O": 0, "I": 1})
    return df