# Probe: round-start score-difference AUC by subset (G-side tomes only).
# Subsets: all rounds; matches with >= 24 rounds; the 40 snapshot matches; 40-match-sized random
# draws of >=24-round matches (seed 0, 200 draws) to show how much a 40-match AUC can move.
import csv, glob, os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")

def files(ch):
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % ch)) if os.path.isdir(x)][0]
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

rend = pq.read_table(files("round_end")[0], columns=["match_key", "round", "winner_team_code"]).to_pandas()
rs = pd.concat([pq.read_table(f, columns=["match_key", "round", "t_score", "ct_score"],
                              filters=[("event_type", "==", "round_freeze_end")]).to_pandas()
                for f in files("round_state")], ignore_index=True).drop_duplicates(["match_key", "round"])
m = rend.merge(rs, on=["match_key", "round"], how="inner")
m = m[m["winner_team_code"].isin([2, 3])].copy()
m["y"] = (m["winner_team_code"] == 3).astype(int)
m["d"] = (m["ct_score"] - m["t_score"]).astype(float)
nr = m.groupby("match_key")["round"].size().rename("n_rounds")
m = m.merge(nr, left_on="match_key", right_index=True)
chosen = sorted({r["match_key"] for r in csv.DictReader(open(os.path.join(ROOT, "logs", "eda", "economy_snapshot_matches.csv"), encoding="utf-8"))})

def auc(df):
    return roc_auc_score(df["y"], df["d"]) if df["y"].nunique() > 1 else float("nan")

print("all rounds            matches %4d rounds %6d AUC %.4f" % (m["match_key"].nunique(), len(m), auc(m)))
s24 = m[m["n_rounds"] >= 24]
print(">=24-round matches    matches %4d rounds %6d AUC %.4f" % (s24["match_key"].nunique(), len(s24), auc(s24)))
s40 = m[m["match_key"].isin(chosen)]
print("snapshot 40 matches   matches %4d rounds %6d AUC %.4f" % (s40["match_key"].nunique(), len(s40), auc(s40)))
keys24 = sorted(s24["match_key"].unique())
rng = np.random.default_rng(0)
aucs = []
g = {k: v for k, v in s24.groupby("match_key")}
for _ in range(200):
    pick = rng.choice(keys24, size=40, replace=False)
    sub = pd.concat([g[k] for k in pick])
    aucs.append(auc(sub))
aucs = np.array(aucs)
print("random 40 of >=24-round matches (200 draws): mean AUC %.4f sd %.4f 5th %.4f 95th %.4f" % (
    aucs.mean(), aucs.std(), np.percentile(aucs, 5), np.percentile(aucs, 95)))
print("draws with AUC < 0.45: %d of 200" % int((aucs < 0.45).sum()))
