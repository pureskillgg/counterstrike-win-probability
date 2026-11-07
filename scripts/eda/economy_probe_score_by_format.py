# Probe: round-start score-difference AUC by match format (header max_rounds) and by match length.
# G-side tomes only: header, round_end, round_state (freeze rows).
import glob, os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")

def files(ch):
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % ch)) if os.path.isdir(x)][0]
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

hdr = pq.read_table(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "max_rounds", "is_wingman"]).to_pandas()
hdr = hdr.rename(columns={"key": "match_key"})
rend = pq.read_table(files("round_end")[0], columns=["match_key", "round", "winner_team_code"]).to_pandas()
rs = pd.concat([pq.read_table(f, columns=["match_key", "round", "t_score", "ct_score"],
                              filters=[("event_type", "==", "round_freeze_end")]).to_pandas()
                for f in files("round_state")], ignore_index=True).drop_duplicates(["match_key", "round"])
m = rend.merge(rs, on=["match_key", "round"]).merge(hdr, on="match_key")
m = m[m["winner_team_code"].isin([2, 3]) & (~m["is_wingman"])].copy()
m["y"] = (m["winner_team_code"] == 3).astype(int)
m["d"] = (m["ct_score"] - m["t_score"]).astype(float)
n = m.groupby("match_key")["round"].size().rename("n_rounds")
m = m.merge(n, left_on="match_key", right_index=True)
m["complete_len"] = m["n_rounds"] >= m["max_rounds"]  # played to the regulation length
def line(lbl, s):
    a = roc_auc_score(s["y"], s["d"]) if s["y"].nunique() > 1 else float("nan")
    print("%-42s matches %5d rounds %6d CT win %.3f AUC %.4f" % (lbl, s["match_key"].nunique(), len(s), s["y"].mean(), a))
for mr in sorted(m["max_rounds"].unique()):
    sub = m[m["max_rounds"] == mr]
    line("max_rounds %d, all lengths" % mr, sub)
    line("max_rounds %d, played full length" % mr, sub[sub["complete_len"]])
    line("max_rounds %d, ended early" % mr, sub[~sub["complete_len"]])
