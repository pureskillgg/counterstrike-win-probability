"""Verifier for D5-8, final header table: per class, how many complete regime-A matches are explained by header totals
under (a) recorded winners, (b) the claimed flip set, with and without allowing the header team to be the complement
(team-identity ambiguity). Also tests the overtime side rule for class X (is there a 24->25 swap?).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_final.py
"""
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
CC = CL.filter(pl.col("complete"))
PC = R.filter(pl.col("regA")).join(CC.select(["match_key", "cls", "hdr_t", "n_rounds"]), on="match_key", how="inner").sort(["match_key", "round"])
data = []
for mk, grp in PC.group_by("match_key", maintain_order=True):
    g = grp.sort("round")
    data.append((mk[0], g["cls"][0], g["round"].to_numpy(), g["winner"].to_numpy(), int(g["hdr_t"][0]), int(g["n_rounds"][0])))


def side_x(rnd, ot_swap):
    # standard MR12 side of starting-T team; ot_swap=True: T for 25-27 (swap at 24->25), False: CT for 25-27
    base = np.where(rnd <= 12, 2, 3)
    ot = np.where(((rnd - 25) % 6) < 3, 2, 3)
    ot = ot if ot_swap else np.where(ot == 2, 3, 2)
    return np.where(rnd >= 25, ot, base)


def score(cls_name, flips, ot_swap, allow_complement):
    ok = 0
    n = 0
    for mk, c, rnd, win, ht, nr in data:
        if c != cls_name:
            continue
        n += 1
        wf = np.where(np.isin(rnd, flips), 5 - win, win)
        X = int(np.sum(side_x(rnd, ot_swap) == wf))
        good = (X == ht) or (allow_complement and (nr - X == ht))
        ok += int(good)
    return ok, n


print("A. header agreement by class (complete regime-A late matches):")
for c, flips in [("none", []), ("X", [13, 25, 28]), ("X", [13]), ("Y", [9]), ("Y", [])]:
    for allow in [False, True]:
        ok, n = score(c, flips, True, allow)
        print(f"   class {c:4s} flips {str(flips):14s} complement_ok={str(allow):5s}: {ok} / {n}")

print("\nB. overtime side rule for class X (flips 13 fixed, vary 25/28 flips and OT rule):")
for ot_swap in [True, False]:
    for flips in [[13], [13, 25], [13, 28], [13, 25, 28]]:
        ok, n = score("X", flips, ot_swap, False)
        print(f"   OT swap at 24->25 = {str(ot_swap):5s} flips {str(flips):16s}: {ok} / {n} exact")

print("\nC. Y matches: complement pattern check (residual under flip 9 must be complement):")
res_all = 0
res_comp = 0
for mk, c, rnd, win, ht, nr in data:
    if c != "Y":
        continue
    wf = np.where(rnd == 9, 5 - win, win)
    X = int(np.sum(side_x(rnd, True) * 0 + np.where(rnd <= 8, 2, 3) == wf))
    if X != ht:
        res_all += 1
        res_comp += int(nr - X == ht)
print(f"   residual matches: {res_all}; of which header equals the complement total: {res_comp}")
