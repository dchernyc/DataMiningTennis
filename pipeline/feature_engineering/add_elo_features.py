"""
add_elo_features.py
-------------------
Replicates the Elo logic https://tennisabstract.com/reports/atp_elo_ratings.html and adds four columns:
    w_elo_before  – winner's Elo rating BEFORE the match
    l_elo_before  – loser's  Elo rating BEFORE the match
    w_elo_after   – winner's Elo rating AFTER  the match
    l_elo_after   – loser's  Elo rating AFTER  the match

Infos:
  - Starting rating:    1500
  - K-factor:           250 / (matches_played + 5) ^ 0.4
  - Grand Slam bonus:   k *= 1.1  when tourney_level == "G"
  - Expected score:     1 / (1 + 10 ^ ((opponent_rating - own_rating) / 400))
  - Update:             new_rating = old_rating + k * (actual - expected)
"""

import pandas as pd

START_ELO = 1500.0
GRAND_SLAM_BONUS = 1.1
BLEND_RATIO = 0.5

def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not {"w_id", "l_id", "tourney_level", "surface"}.issubset(df.columns):
        raise ValueError("Input DataFrame must contain 'w_id', 'l_id', 'tourney_level', and 'surface' columns.")

    if not df["tourney_level"].eq("G").any():
        raise ValueError("No Grand Slam matches found. Ensure 'tourney_level' has 'G' for Grand Slams.")

    # State
    overall_elo:    dict[str, float]             = {}
    surface_elo:    dict[str, dict[str, float]]  = {}
    overall_played: dict[str, int]               = {}
    surface_played: dict[str, dict[str, int]]    = {}
 
    def _get_overall(player: str) -> float:
        return overall_elo.get(player, START_ELO)
 
    def _get_surface(player: str, surface: str) -> float:
        return surface_elo.get(player, {}).get(surface, START_ELO)
 
    def _k(n_played: int, is_grand_slam: bool) -> float:
        base = 250.0 / ((n_played + 5) ** 0.4)
        return base * GRAND_SLAM_BONUS if is_grand_slam else base
 
    # Output lists
    cols: dict[str, list[float]] = {
        "w_elo_before":                 [],
        "l_elo_before":                 [],
        "w_elo_after":                  [],
        "l_elo_after":                  [],
        "w_elo_surface_before":         [],
        "l_elo_surface_before":         [],
        "w_elo_surface_after":          [],
        "l_elo_surface_after":          [],
        "w_elo_surface_blended_before": [],
        "l_elo_surface_blended_before": [],
    }
 
    for _, row in df.iterrows():
        w      = row["w_id"]
        l      = row["l_id"]
        surf   = row["surface"]
        is_gs  = str(row["tourney_level"]).strip() == "G"
 
        # Current ratings (before this match)
        rW_overall  = _get_overall(w)
        rL_overall  = _get_overall(l)
        rW_surf     = _get_surface(w, surf)
        rL_surf     = _get_surface(l, surf)

        # Blended surface ratings
        rW_blended  = BLEND_RATIO * rW_surf + (1 - BLEND_RATIO) * rW_overall
        rL_blended  = BLEND_RATIO * rL_surf + (1 - BLEND_RATIO) * rL_overall
 
        # Snapshot ratings before the match
        cols["w_elo_before"].append(round(rW_overall, 4))
        cols["l_elo_before"].append(round(rL_overall, 4))
        cols["w_elo_surface_before"].append(round(rW_surf, 4))
        cols["l_elo_surface_before"].append(round(rL_surf, 4))
        cols["w_elo_surface_blended_before"].append(round(rW_blended, 4))
        cols["l_elo_surface_blended_before"].append(round(rL_blended, 4))
 
        # Overall Elo update
        eW_o = 1.0 / (1.0 + 10.0 ** ((rL_overall - rW_overall) / 400.0))
        eL_o = 1.0 - eW_o
        kW_o = _k(overall_played.get(w, 0), is_gs)
        kL_o = _k(overall_played.get(l, 0), is_gs)
        rW_overall_new = rW_overall + kW_o * (1.0 - eW_o)
        rL_overall_new = rL_overall + kL_o * (0.0 - eL_o)
 
        overall_elo[w] = rW_overall_new
        overall_elo[l] = rL_overall_new
 
        cols["w_elo_after"].append(round(rW_overall_new, 4))
        cols["l_elo_after"].append(round(rL_overall_new, 4))

        # Surface Elo update
        eW_s = 1.0 / (1.0 + 10.0 ** ((rL_surf - rW_surf) / 400.0))
        eL_s = 1.0 - eW_s
        kW_s = _k(surface_played.get(w, {}).get(surf, 0), is_gs)
        kL_s = _k(surface_played.get(l, {}).get(surf, 0), is_gs)
        rW_surf_new = rW_surf + kW_s * (1.0 - eW_s)
        rL_surf_new = rL_surf + kL_s * (0.0 - eL_s)
 
        surface_elo.setdefault(w, {})[surf] = rW_surf_new
        surface_elo.setdefault(l, {})[surf] = rL_surf_new
 
        cols["w_elo_surface_after"].append(round(rW_surf_new, 4))
        cols["l_elo_surface_after"].append(round(rL_surf_new, 4))

        # Increment match counter
        overall_played[w] = overall_played.get(w, 0) + 1
        overall_played[l] = overall_played.get(l, 0) + 1
        surface_played.setdefault(w, {})[surf] = surface_played.get(w, {}).get(surf, 0) + 1
        surface_played.setdefault(l, {})[surf] = surface_played.get(l, {}).get(surf, 0) + 1
 
    # Attach columns
    for col_name, values in cols.items():
        df[col_name] = values
 
    return df