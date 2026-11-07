"""Verifier for D5-8, class-Y extension: can any 1-2 flipped rounds (1..16) with swap at 8 (and 8,16) explain all class-Y
complete matches' header totals? Also reports the residual for the best single-flip set.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_yflip.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

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
Yc = CL.filter((pl.col("cls") == "Y") & pl.col("complete")).select("match_key")
PY = R.filter(pl.col("regA")).join(Yc, on="match_key", how="inner").sort(["match_key", "round"])
data = []
for mk, grp in PY.group_by("match_key", maintain_order=True):
    g = grp.sort("round")
    data.append((mk[0], g["round"].to_numpy(), g["winner"].to_numpy(), int(g["t_starters_score_final"][0])))
print("class-Y complete matches:", len(data))
print("n_rounds distribution:", sorted({len(d[1]) for d in data}))

results = []
for S in [(8,), (8, 16)]:
    for nf in range(0, 3):
        for F in itertools.combinations(range(1, 17), nf):
            ok = 0
            for mk, rnd, win, ht in data:
                sb = np.zeros_like(rnd)
                for s in S:
                    sb = sb + (rnd > s).astype(int)
                side_x = np.where(sb % 2 == 0, 2, 3)
                wf = np.where(np.isin(rnd, list(F)), 5 - win, win)
                ok += int(int(np.sum(side_x == wf)) == ht)
            results.append((ok, S, F))
results.sort(key=lambda t: -t[0])
print("best (exact, swaps, flips) over swap sets {8} and {8,16} with up to 2 flips in rounds 1..16:")
for t in results[:10]:
    print("   ", t)
# baseline: swap at 8, no flips, strict and complement-allowed
for F0 in [(), (9,)]:
    strict = comp = 0
    for mk, rnd, win, ht in data:
        sb = (rnd > 8).astype(int)
        side_x = np.where(sb % 2 == 0, 2, 3)
        wf = np.where(np.isin(rnd, list(F0)), 5 - win, win)
        X = int(np.sum(side_x == wf))
        strict += int(X == ht)
        comp += int(X == ht or (len(rnd) - X == ht))
    print(f"baseline swap (8,) flips {F0}: strict {strict}/{len(data)}, complement-allowed {comp}/{len(data)}")
# residual for swaps (8,), flip (9,)
S, F = (8,), (9,)
resid = []
for mk, rnd, win, ht in data:
    sb = np.zeros_like(rnd)
    for s in S:
        sb = sb + (rnd > s).astype(int)
    side_x = np.where(sb % 2 == 0, 2, 3)
    wf = np.where(np.isin(rnd, list(F)), 5 - win, win)
    X = int(np.sum(side_x == wf))
    if X != ht:
        resid.append((mk.split('/')[-1][:8], len(rnd), X, ht))
print(f"\nresidual under swaps (8,) flip (9,): {len(resid)} matches; first 12:")
for r in resid[:12]:
    print("   ", r)
