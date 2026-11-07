"""verify D5-9, step 10: reproduce the claim's joint table (1,621 13-only; 161 9-only; none both) under the
population the claim most likely used: late regime-A matches EXCLUDING the e2-affected set, with the loose tie rule
(ambiguous boundary takes the no-swap-fit truth when that fit holds, else the swap-fit truth).

Also: the 2026 header fields is_wingman and max_rounds, cross-tabbed with the 4-player set (format check).

e2-affected (as in scripts/swarm/e2_side_check_r13.py): matches with a round-13 bomb outcome (defused or exploded) whose
round_end winner disagrees, restricted to round_end player_count == 20.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_10_joint_affected.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_9")
R = pl.read_parquet(SCR / "rounds.parquet")
M = pl.read_parquet(SCR / "matches.parquet")
meta = {r["match_key"]: r for r in M.to_dicts()}
LATE = {f"{d:02d}" for d in range(12, 26)}

# e2-affected set
r13 = R.filter((pl.col("round") == 13) & (pl.col("end_winner").is_not_null()) &
               ((pl.col("n_defused") + pl.col("n_exploded")) > 0) & (pl.col("end_player_count") == 20))
r13 = r13.with_columns(bomb_side=pl.when(pl.col("n_defused") > 0).then(pl.lit(3)).otherwise(pl.lit(2)))
affected = set(r13.filter(pl.col("bomb_side") != pl.col("end_winner"))["match_key"].to_list())
print("e2-affected matches (recomputed):", len(affected), "(e2 log: 434)")

fe = defaultdict(dict)
rend = defaultdict(dict)
for r in R.iter_rows(named=True):
    if r["fe_t"] is not None:
        fe[r["match_key"]][r["round"]] = (r["fe_t"], r["fe_ct"])
    if r["end_winner"] is not None:
        rend[r["match_key"]][r["round"]] = r["end_winner"]


def fit(mk, r):
    F = fe[mk]
    if r not in F or r + 1 not in F:
        return ("none", None, None)
    t0, c0 = F[r]
    t1, c1 = F[r + 1]
    dt, dc = t1 - t0, c1 - c0
    s = (dt + dc) == 1
    ns = s and dt in (0, 1) and dc in (0, 1)
    sw = s and (t1 - c0) in (0, 1) and (c1 - t0) in (0, 1)
    tn = (2 if dt == 1 else 3) if ns else None
    ts = (2 if (c1 - t0) == 1 else 3) if sw else None
    if ns and sw:
        return ("ambig", tn, ts)
    if ns:
        return ("noswap", tn, None)
    if sw:
        return ("swap", None, ts)
    return ("bad", None, None)


def truth_loose(mk, r):
    _, tn, ts = fit(mk, r)
    return tn if tn is not None else ts


def truth_strict(mk, r):
    cls, tn, ts = fit(mk, r)
    if cls == "noswap":
        return tn
    if cls == "swap":
        return ts
    return None


late_A = [mk for mk in meta if meta[mk]["regime"] == "A" and meta[mk]["day"] in LATE]
for name, pop in [("late regime A, all", late_A),
                  ("late regime A, excluding e2-affected", [m for m in late_A if m not in affected])]:
    for rule, tf in [("loose", truth_loose), ("strict", truth_strict)]:
        joint = Counter()
        for mk in pop:
            t9 = tf(mk, 9)
            t13 = tf(mk, 13)
            if t9 is None or t13 is None or 9 not in rend[mk] or 13 not in rend[mk]:
                joint["not both usable"] += 1
                continue
            joint[("13 only" if (rend[mk][13] != t13 and rend[mk][9] == t9) else
                   "9 only" if (rend[mk][9] != t9 and rend[mk][13] == t13) else
                   "both" if (rend[mk][9] != t9 and rend[mk][13] != t13) else "neither")] += 1
        print(f"{name:<40} rule={rule:<6} {dict(sorted(joint.items()))}")
print("claim: 13 only 1,621; 9 only 161; both 0")

# 2026 header format fields
print("\n2026 header format fields")
hf = sorted(glob.glob(str(Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\header") / "dataframe_*")))[0]
H = pl.read_parquet(hf)
cols = [c for c in H.columns if c in ("is_wingman", "max_rounds", "key", "match_type", "map_name")]
print("  header columns present:", cols)
if "is_wingman" in H.columns and "max_rounds" in H.columns:
    print("  crosstab is_wingman x max_rounds:", H.group_by(["is_wingman", "max_rounds"]).len().sort(["is_wingman", "max_rounds"]).to_dicts())
    # 4-player set from the 2026 build (same rule as verify_D5_9_07_2026.py: distinct player_id in player_info)
    BASE = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
    PIf = sorted(glob.glob(str(BASE / "basic_player_info.2026-09-18,2026-10-08" / "dataframe_*")))[0]
    PI = pl.read_parquet(PIf, columns=["match_key", "player_id"])
    npl = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n"))
    H2 = H.rename({"key": "match_key"}).join(npl, on="match_key", how="left")
    print("  crosstab is_wingman x n_players(4):",
          H2.with_columns(is4=pl.col("n") == 4).group_by(["is_wingman", "is4"]).len().sort(["is_wingman", "is4"]).to_dicts())
print("\nDONE step 10")
