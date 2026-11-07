"""verify D1-3: raw vs tome for bomb_state and round_state (one affected match, read-only).

Match: e025c2e1-2251-4c9a-90be-a68cab42d04d (day 12). Compares row counts and round-13 bomb events and
round-13 score fields between the raw E: match folder and the tome.
"""
import glob
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
MID = "e025c2e1-2251-4c9a-90be-a68cab42d04d"
RAWM = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12") / MID
TOMEB = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def tome(ch, cols):
    fs = sorted(glob.glob(str(TOMEB / f"basic_{ch}{SUF}" / "dataframe_*")))
    df = pl.concat([pl.read_parquet(f, columns=cols + ["match_key"]) for f in fs], how="vertical_relaxed")
    return df.filter(pl.col("match_key").str.contains(MID))


for ch, cols in [("bomb_state", ["round", "tick", "event_type", "player_team_code"]),
                 ("round_state", ["round", "tick", "event_type", "t_score", "ct_score"])]:
    rawf = RAWM / ch
    print(f"== {ch}: raw exists={rawf.exists()}")
    if not rawf.exists():
        continue
    raw = pl.read_parquet(str(rawf))
    print("raw rows", raw.height, " columns:", [c for c in cols if c in raw.columns], "missing:", [c for c in cols if c not in raw.columns])
    t = tome(ch, cols)
    print("tome rows", t.height)
    if ch == "bomb_state":
        for name, df in [("raw", raw), ("tome", t)]:
            d = df.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]) & (pl.col("round").is_between(12, 14)))
            print(name, "bomb events r12-14:", d.select(["round", "tick", "event_type", "player_team_code"]).sort(["round", "tick"]).rows())
    else:
        for name, df in [("raw", raw), ("tome", t)]:
            d = df.filter(pl.col("round") == 13).sort("tick")
            first, last = d.row(0), d.row(d.height - 1)
            print(name, "round 13 first row:", first[:5] if len(first) >= 5 else first, " last row:", last)
            d2 = df.filter(pl.col("event_type") == "round_officially_ended").sort("tick")
            print(name, "round_officially_ended rows (round, tick, t, ct):", d2.select(["round", "tick", "t_score", "ct_score"]).rows()[10:16])
