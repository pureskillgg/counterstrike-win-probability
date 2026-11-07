r"""D4 diagnostic: for 2024 complete regulation matches that misfit the S12 header test with round_end winners,
which single round (if its winner were flipped) would reconcile the header totals? Also the size of the misfit.
Affected = >= 1 label-based round-13 mismatch (flip candidates there come from bomb rows).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_single_flip.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"])
Hd = {r["key"]: r for r in H.iter_rows(named=True)}
re_ = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = set(R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
           .filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
by = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
diffs = {}
fixers = {}
n_mis = 0
n_fixed_by_one = 0
n_total = 0
for m, rounds in by.items():
    h = Hd.get(m)
    if h is None or m in mm13:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    n_total += 1
    wa = 0
    sides = []
    for x in rounds:
        side_A = T if int(x["round"]) <= 12 else CT
        isA = int(x["winner_team_code"]) == side_A
        sides.append(isA)
        wa += isA
    if (wa, n - wa) == (tA, tB):
        continue
    n_mis += 1
    d = wa - tA
    diffs[d] = diffs.get(d, 0) + 1
    fix_rounds = []
    for i, x in enumerate(rounds):
        new_wa = wa - (1 if sides[i] else 0) + (0 if sides[i] else 1)
        if (new_wa, n - new_wa) == (tA, tB):
            fix_rounds.append(int(x["round"]))
    if fix_rounds:
        n_fixed_by_one += 1
        for r in fix_rounds:
            fixers[r] = fixers.get(r, 0) + 1
print(f"unaffected complete regulation matches: {n_total}; S12 misfits with round_end winners: {n_mis}")
print("misfit size (wa - header t_starters), counts:", dict(sorted(diffs.items())))
print(f"misfits fixable by flipping exactly one round: {n_fixed_by_one}")
print("round number that, flipped alone, reconciles (count of matches; multiple rounds can count):")
print(dict(sorted(fixers.items())))
