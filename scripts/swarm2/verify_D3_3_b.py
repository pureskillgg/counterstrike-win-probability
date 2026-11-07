"""verify_D3_3 part B: score-only swap evidence per boundary, by per-match format (no winner codes used).

For each boundary r -> r+1 (freeze rows of r and r+1 both present), classify the freeze-score change:
  noswap  : side scores grew by one for the winner, no crossover  (dT,dC) in {(1,0),(0,1)}
  swap    : crossover (t' = c + [CT won], c' = t + [T won])        (xT,xC) in {(1,0),(0,1)}
  ambig   : both fit (happens when t == c or near ties); neither : fits neither
Reports, per format group (announce-rule MR 16 or 24), the class counts at the rounds of interest, and the
first unambiguous swap boundary per match. ASSUMPTION: announce-rule MR is only used as a grouping key here.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect().sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
F = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    F[m][int(r)] = (int(t), int(c))


def cls(t, c, t2, c2):
    dT, dC = t2 - t, c2 - c
    xT, xC = t2 - c, c2 - t
    nos = dT in (0, 1) and dC in (0, 1) and dT + dC == 1
    swp = xT in (0, 1) and xC in (0, 1) and xT + xC == 1
    if nos and swp:
        return "ambig"
    if nos:
        return "noswap"
    if swp:
        return "swap"
    return "neither"


def day(m):
    return int(m.split("/")[3])


ROUNDS = [7, 8, 9, 10, 12, 13, 14, 19, 24, 25, 26, 27, 28, 29]
groups = {"MR16": [m for m in F if MR.get(m, 24) == 16], "MR24": [m for m in F if MR.get(m, 24) == 24]}
for g, ms in groups.items():
    for lab, sel in (("d01_07", lambda m: day(m) <= 7), ("d12_25", lambda m: day(m) >= 12)):
        mm = [m for m in ms if sel(m)]
        print(f"\n== {g} {lab}: matches {len(mm)} ==")
        print("  round: noswap / swap / ambig / neither  (boundary r -> r+1)")
        for r in ROUNDS:
            c_ = collections.Counter()
            for m in mm:
                if r in F[m] and (r + 1) in F[m]:
                    t, c = F[m][r]
                    t2, c2 = F[m][r + 1]
                    c_[cls(t, c, t2, c2)] += 1
            n = sum(c_.values())
            if n:
                print(f"   r={r:>2}: noswap {c_['noswap']:>5} swap {c_['swap']:>5} ambig {c_['ambig']:>5} neither {c_['neither']:>4}  (n={n})")
        # first unambiguous swap boundary per match
        first_swap = collections.Counter()
        for m in mm:
            fs = None
            for r in sorted(F[m]):
                if r + 1 in F[m]:
                    t, c = F[m][r]
                    t2, c2 = F[m][r + 1]
                    if cls(t, c, t2, c2) == "swap":
                        fs = r
                        break
            first_swap[fs] += 1
        print("  first unambiguous swap boundary (round r, boundary r->r+1) per match:", dict(sorted(first_swap.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))))
