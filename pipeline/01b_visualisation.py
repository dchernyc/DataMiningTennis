#Dataset visualisation
#Outputs are saved into root/data/visualisation/ (files are already uploaded)

import pandas as pd
import os
import matplotlib.pyplot as plt
from config import CLEANED_DATA_PATH, VISUALISATION_DIR

pd.set_option('display.max_columns', None)
df = pd.read_csv(CLEANED_DATA_PATH) 

#Create folder for plots
if not os.path.exists(VISUALISATION_DIR):
    os.makedirs(VISUALISATION_DIR)

#Surface distribution: pie chart
counts_surface = df["surface"].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(
    counts_surface,
    labels=df["surface"].unique(),
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Surface Type Distribution")

surface_plot = VISUALISATION_DIR / "surface_distribution.png"

plt.savefig(surface_plot, bbox_inches="tight")
plt.close()
print(f"Saved: {surface_plot}")

#Tournament level distribution: bar plot
levels = {
    "G": "Grand Slam",
    "M": "Masters 1000",
    "500": "ATP 500",
    "250": "ATP 250",
    "F": "ATP Finals",
    "D": "Davis Cup",
    "A": "ATP Tour",
    "O": "Other"
}

level_counts = df["tourney_level"].map(levels).value_counts()

plt.figure(figsize=(10, 6))
level_counts.plot(kind="bar")
plt.xlabel("Tournament Level")
plt.ylabel("Count")
plt.title("Tournament Level Distribution")
plt.xticks(rotation=45)

level_plot = VISUALISATION_DIR / "tournament_levels.png"

plt.savefig(level_plot, bbox_inches="tight")
plt.close()
print(f"Saved: {level_plot}")

#Court type distribution: pie chart
court_types = {
    "I": "Indoor",
    "O": "Outdoor"
}
court_counts = df["indoor"].map(court_types).value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    court_counts,
    labels=court_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Court Type Distribution")

court_plot = VISUALISATION_DIR / "court_type_distribution.png"

plt.savefig(court_plot, bbox_inches="tight")
plt.close()
print(f"Saved: {court_plot}")

#Correlation matrix for selected numeric columns
corr_features = [
    "w_age",
    "l_age",
    "w_ht",
    "l_ht",
    "w_rank",
    "l_rank",
    "w_rank_points",
    "l_rank_points",
    "minutes",
    "w_ace",
    "l_ace"
]

df_corr = df[corr_features].corr()
plt.figure(figsize=(12, 10))
plt.imshow(df_corr, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_features)), corr_features, rotation=90)
plt.yticks(range(len(corr_features)), corr_features)
plt.title("Correlation Matrix")

corr_plot = VISUALISATION_DIR / "correlation_matrix.png"

plt.savefig(corr_plot, bbox_inches="tight")
plt.close()
print(f"Saved: {corr_plot}")