# Probe: which round is the first side swap in the sampled matches (G-side tomes only).
# Team identity is tracked with player_id_fixed sets from basic_player_info; the swap is the first
# round in which the team on CT differs from the team on CT in round 1.
import csv, glob, os
from collections import Counter
import pyarrow.parquet as pq
import pandas as pd

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
keys = sorted({r["match_key"] for r in csv.DictReader(open(os.path.join(ROOT, "logs", "eda", "economy_snapshot_matches.csv"), encoding="utf-8"))})
def files(ch):
    d = [x for x in glob.glob(os.path.join(TOME, "basic_%s.*" % ch)) if os.path.isdir(x)][0]
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))
hdr = pq.read_table(os.path.join(TOME, "header", "dataframe_00000"), columns=["key", "max_rounds"]).to_pandas().set_index("key")
rend = pq.read_table(files("round_end")[0], columns=["match_key", "round"], filters=[("match_key", "in", keys)]).to_pandas()
pi = pq.read_table(files("player_info")[0], columns=["match_key", "round", "player_id_fixed", "team_code"],
                   filters=[("match_key", "in", keys)]).to_pandas()
pi = pi[pi["team_code"].isin([2, 3]) & pi["player_id_fixed"].notna()]
flip = Counter(); maxr = Counter()
for mk in keys:
    rounds = sorted(rend.loc[rend["match_key"] == mk, "round"].tolist())
    p = pi[pi["match_key"] == mk]
    A = None; first_ct = None; flip_round = None
    for r in rounds:
        pr = p[p["round"] == r]
        T = set(pr.loc[pr["team_code"] == 2, "player_id_fixed"].astype("int64"))
        C = set(pr.loc[pr["team_code"] == 3, "player_id_fixed"].astype("int64"))
        if A is None:
            if not T or not C: continue
            A = frozenset(T); first_ct = frozenset(C); continue
        # team on CT in round 1 is 'first_ct'; in round r it is first_ct if its players are on CT
        same = len(C & set(first_ct)) >= len(C & set(A))
        if not same and flip_round is None:
            flip_round = r
    mr = int(hdr.loc[mk, "max_rounds"]); maxr[mr] += 1
    flip[(mr, flip_round)] += 1
for k in sorted(flip, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
    print("max_rounds", k[0], "first CT-team flip at round", k[1], "matches", flip[k])
