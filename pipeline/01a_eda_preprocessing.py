#TennisMyLife dataset exploratory data analysis and data cleaning/preprocessing

import pandas as pd
from config import RAW_DATA_PATH, CLEANED_DATA_PATH

#Load raw dataset
pd.set_option('display.max_columns', None)
df = pd.read_csv(RAW_DATA_PATH) 
#DtypeWarning: Columns (3,6,7,9,13,15,16,17,19,23,25,26,27,29,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49) have mixed types.
# -> only columns that should be numeric have mixed types

print("\n"+"Shape of Dataframe:", df.shape)
print("\n"+"Column information:")
df.info()

#Drop columns with more than 50% missing entries
print("\n"+"Missing values per column:")
print(df.isnull().sum().sort_values(ascending=False))
df = df.drop(columns=["winner_seed", "winner_entry", "loser_seed", "loser_entry"])

#Rename columns winner_... and loser_... to w_... and l_... for further preprocessing
df = df.rename(columns={
    col : col.replace("winner_", "w_").replace("loser_", "l_")
    for col in df.columns
})

numeric_columns = [
    'draw_size', 'tourney_date', 'match_num', 'w_ht', 'w_age', 'w_rank', 'w_rank_points', 'l_ht',
       'l_age', 'l_rank', 'l_rank_points', 'best_of', 'minutes', 'w_ace', 'w_df',
       'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved',
       'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon',
       'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]


#Adjust dtype in all numeric columns and remove invalid rows
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=numeric_columns, how="all")
print("\n"+"Shape of Dataframe after removing wrong dtypes:", df.shape)

#Drop all rows which contain at least one missing value
df = df.dropna().reset_index(drop=True)

#Check if there are duplicate rows in the dataset
print("\n"+"Number of duplicate rows:", df.duplicated().sum())

print("\n"+"Column information for cleaned dataset:")
df.info()

#Statistical analysis -> check if value ranges of columns are valid by looking at min/max values
print("\n"+"Statistical analysis:",) 
print(df.describe(include="all"))

#Outlier detection: Interquartile range
print("\n"+"Outlier detection:")
for col in numeric_columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: Lower Bound: {lower_bound}, Upper Bound: {upper_bound}, Outliers: {len(outliers)}")

#Create cleaned csv file for further preprocessing (file already uploaded)
#df.to_csv(CLEANED_DATA_PATH, index=False)
