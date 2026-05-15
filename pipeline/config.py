"""
Shared configuration for all models.
Import this in every script to keep paths and constants consistent.
"""

from pathlib import Path

# ──Directory paths ───────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "data" 
VISUALISATION_DIR = ROOT_DIR / "data" / "visualisation"

TRAIN_FILE = DATASET_DIR / "train.csv"
TEST_FILE = DATASET_DIR / "test.csv"
RAW_DATA_PATH = DATASET_DIR / "2000_2026_raw.csv"
CLEANED_DATA_PATH = DATASET_DIR / "2000_2026_cleaned.csv"


# ──Model features ───────────────────────────────────────────────────────────

FEATURES = [
    'ht_diff',
    'age_diff',
    'rank_diff',
    'rank_points_diff',
    'ace_rate_10_diff',
    'df_rate_10_diff',
    'bp_save_rate_10_diff',
    'serve_win_rate_10_diff',
    '1stWon_rate_10_diff',
    '2ndWon_rate_10_diff',
    'h2h_diff',
    'win_rate_10_diff',
    'elo_before_diff',
    'elo_surface_before_diff',
    'elo_surface_blended_before_diff',
    'surface_Carpet',
    'surface_Clay',
    'surface_Grass',
    'surface_Hard',
    'tourney_level_250', 
    'tourney_level_500', 
    'tourney_level_A',
    'tourney_level_D', 
    'tourney_level_F', 
    'tourney_level_G',
    'tourney_level_M', 
    'tourney_level_O'
]