# Probe: sign and size of the round-start score difference across ALL matches (G-side tomes only).
# AUC of (ct_score - t_score at round_freeze_end) for ct_win, using the score fields as stored.
import glob, os
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")

def files(ch):
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % ch)) if os.path.isdir(x)][0]
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

import pandas as pd
rend = pq.read_table(files("round_end")[0], columns=["match_key", "round", "winner_team_code"]).to_pandas()
rs = pd.concat([pq.read_table(f, columns=["match_key", "round", "t_score", "ct_score"],
                              filters=[("event_type", "==", "round_freeze_end")]).to_pandas()
                for f in files("round_state")], ignore_index=True)
rs = rs.drop_duplicates(["match_key", "round"])
m = rend.merge(rs, on=["match_key", "round"], how="inner")
m = m[m["winner_team_code"].isin([2, 3])]
y = (m["winner_team_code"] == 3).astype(int).to_numpy()
d = (m["ct_score"] - m["t_score"]).to_numpy(dtype=float)
print("rounds with freeze row and winner:", len(m), "matches:", m["match_key"].nunique())
print("CT win rate:", round(y.mean(), 4))
print("AUC ct_score - t_score vs ct_win:", round(roc_auc_score(y, d), 4))
for lead in [-2, -1, 0, 1, 2]:
    sel = d == lead
    print("diff", lead, "rounds", int(sel.sum()), "CT win rate", round(y[sel].mean(), 4) if sel.any() else None)
