"""D3-5 verifier, step P (2024 G: tome): is the rising round-13 mismatch by day explained by the share of contiguous
matches (round 1 recorded) on each day? Per day 12..25: contiguous vs round-1-missing counts, and for MR12 matches
the number of valid round-13 deltas that disagree with the round_end winner, split by class. Read only.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
sec = RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("h"))
hmap = {m: h for m, h in sec.iter_rows()}
W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
F = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    F[m][r] = (t, c)
day = lambda m: int(m.split("/")[3])
cls = lambda m: "contig" if sorted(W[m]) == list(range(1, len(W[m]) + 1)) else "start2"

per = collections.defaultdict(collections.Counter)
for m in W:
    d = day(m)
    if d < 12:
        continue
    per[d][cls(m) + "_matches"] += 1
    if hmap.get(m) != 12:
        continue
    r = 13
    if r in F[m] and (r + 1) in F[m]:
        t1, c1 = F[m][r]
        t2, c2 = F[m][r + 1]
        dd = (t2 - t1, c2 - c1)
        if dd in [(1, 0), (0, 1)]:
            D = T if dd == (1, 0) else CT
            per[d][cls(m) + "_r13_valid"] += 1
            per[d][cls(m) + "_r13_W!=D"] += int(W[m][r] != D)
print(f"{'day':>4} {'contig_m':>8} {'start2_m':>8} {'contig_r13_valid':>16} {'contig_W!=D':>11} "
      f"{'start2_r13_valid':>16} {'start2_W!=D':>11}")
tot = collections.Counter()
for d in sorted(per):
    c = per[d]
    for k, v in c.items():
        tot[k] += v
    print(f"{d:>4} {c['contig_matches']:>8} {c['start2_matches']:>8} {c['contig_r13_valid']:>16} "
          f"{c['contig_r13_W!=D']:>11} {c['start2_r13_valid']:>16} {c['start2_r13_W!=D']:>11}")
print("total:", dict(tot))
print("== end ==")
