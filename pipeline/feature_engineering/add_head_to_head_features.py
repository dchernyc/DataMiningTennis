import pandas as pd
import numpy as np

def add_head_to_head_features(df):
    df = df.copy()
    df = df.sort_values("match_id").reset_index(drop=True)

    h2h = {}
    winner_h2h = []
    loser_h2h = []

    for _, row in df.iterrows():
        w = row['w_id']
        l = row['l_id']

        if w < l:
            key = (w, l)
        else:          
            key = (l, w) 

        if key in h2h:
            wins_p1, total = h2h[key]
            p1 = key[0]

            if w == p1:
                w_rate = wins_p1 / total
            else:
                w_rate = (total - wins_p1) / total
        else:
            # w_rate = 0.5
            w_rate= np.nan

        l_rate = 1 - w_rate

        winner_h2h.append(w_rate)
        loser_h2h.append(l_rate)

        # Update head-to-head record
        # New head-to-head record for this player pair
        if key not in h2h:
            h2h[key] = [0, 0]

        # Increase wins for player 1 if player 1 is the winner
        if w == key[0]:
            h2h[key][0] += 1

        # Increase total matches by 1
        h2h[key][1] += 1

    df['w_h2h'] = winner_h2h
    df['l_h2h'] = loser_h2h
    return df