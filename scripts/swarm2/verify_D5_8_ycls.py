"""Verifier for D5-8, class-Y pass: the side-swap pattern of halftime-8 matches (class Y) and header agreement.

  Y1. round count distribution of complete regime-A matches by class (X, Y, none).
  Y2. boundary classes NS/SW for all boundaries 1..29 in class Y (recorded winners) vs class X and class none.
  Y3. header test for class Y complete matches: brute force over swap sets (boundaries) and flip sets (rounds),
      report the best combination and how many class-Y matches it explains. Uses only recorded winners.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_ycls.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})
LATE = [f"{d:02d}" for d in range(12, 26)]
RA = R.filter(pl.col("regA") & pl.col("day").is_in(LATE))

s9 = RA.filter(pl.col("round") == 9).select(["match_key", ((pl.col("step_side").is_not_null()) & (pl.col("step_side") != pl.col("winner"))).alias("Y")])
s13 = RA.filter(pl.col("round") == 13).select(["match_key", ((pl.col("step_side").is_not_null()) & (pl.col("step_side") != pl.col("winner"))).alias("X")])
CL = RA.select("match_key").unique().join(s9, on="match_key", how="left").join(s13, on="match_key", how="left").with_columns(
    pl.col("Y").fill_null(False), pl.col("X").fill_null(False)).with_columns(
    cls=pl.when(pl.col("X")).then(pl.lit("X")).when(pl.col("Y")).then(pl.lit("Y")).otherwise(pl.lit("none")))
Mx = RA.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("t_starters_score_final").first().alias("hdr_t"),
                                  pl.col("ct_starters_score_final").first().alias("hdr_ct"))
CL = CL.join(Mx, on="match_key", how="left").with_columns(complete=(pl.col("hdr_t") + pl.col("hdr_ct") == pl.col("n_rounds")))
print("Y1. regime-A late: class x complete x n_rounds:")
print(CL.group_by(["cls", "complete"]).len().sort(["cls", "complete"]))
print(CL.filter(pl.col("complete")).group_by(["cls", "n_rounds"]).len().sort(["cls", "n_rounds"]).head(40))

# Y2. boundary classes for all boundaries, by class
FE = R.select(["match_key", "round", "t_fe", "ct_fe", "winner", "regA", "day"])
N = FE.with_columns(nt=pl.col("t_fe").shift(-1).over("match_key"), nct=pl.col("ct_fe").shift(-1).over("match_key"),
                    nr=pl.col("round").shift(-1).over("match_key"))
N = N.filter((pl.col("nr") == pl.col("round") + 1) & pl.col("nt").is_not_null() & pl.col("winner").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
N = N.with_columns(ns=(pl.col("nt") == pl.col("t_fe") + a) & (pl.col("nct") == pl.col("ct_fe") + b),
                   sw=(pl.col("nt") == pl.col("ct_fe") + b) & (pl.col("nct") == pl.col("t_fe") + a))
N = N.with_columns(cls_b=pl.when(pl.col("ns") & pl.col("sw")).then(pl.lit("both"))
                   .when(pl.col("ns")).then(pl.lit("NS")).when(pl.col("sw")).then(pl.lit("SW")).otherwise(pl.lit("neither")))
NA = N.filter(pl.col("regA") & pl.col("day").is_in(LATE)).join(CL.select(["match_key", "cls"]), on="match_key", how="inner")
NA = NA.filter(pl.col("round").is_in(list(range(6, 18)) + [23, 24, 25, 26, 27, 28]))
print("\nY2. regime-A late boundary classes by class (rounds 6-17 and 23-28 boundaries):")
print(NA.filter(pl.col("cls") == "Y").group_by(["round", "cls_b"]).len().sort(["round", "cls_b"]))

# Y3. brute force header test for class Y complete matches
Yc = CL.filter((pl.col("cls") == "Y") & pl.col("complete")).select("match_key")
PY = R.filter(pl.col("regA")).join(Yc, on="match_key", how="inner").sort(["match_key", "round"])
keys = PY["match_key"].unique(maintain_order=True).to_list()
hdr = PY.group_by("match_key", maintain_order=True).agg(pl.col("t_starters_score_final").first().alias("ht")).sort(
    pl.col("match_key").cast(pl.String))
hdr_map = dict(zip(hdr["match_key"].to_list(), hdr["ht"].to_list()))
rounds_map = {}
for mk, grp in PY.group_by("match_key", maintain_order=True):
    g = grp.sort("round")
    rounds_map[mk[0]] = (g["round"].to_numpy(), g["winner"].to_numpy())

SW_CANDS = [8, 12, 16, 20, 24, 27]
FLIP_CANDS = [9, 13, 17, 21, 25, 28]
best = []
for ns in range(0, 4):
    for S in itertools.combinations(SW_CANDS, ns):
        for nf in range(0, 3):
            for F in itertools.combinations(FLIP_CANDS, nf):
                ok = 0
                for mk in keys:
                    rnd, win = rounds_map[mk]
                    swaps_before = np.zeros_like(rnd)
                    for s in S:
                        swaps_before = swaps_before + (rnd > s).astype(int)
                    side_x = np.where(swaps_before % 2 == 0, 2, 3)
                    wf = np.where(np.isin(rnd, list(F)), 5 - win, win)
                    X = int(np.sum(side_x == wf))
                    ok += int(X == hdr_map[mk])
                best.append((ok, S, F))
best.sort(key=lambda t: -t[0])
print(f"\nY3. class-Y complete matches: {len(keys)}; best (exact matches, swap boundaries, flipped rounds):")
for t in best[:8]:
    print("   ", t)
print("   no swaps, no flips:", [t for t in best if t[1] == () and t[2] == ()])
