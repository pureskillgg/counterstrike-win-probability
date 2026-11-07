"""verify D1-6 d01-07 inclusion check (swarm2 verifier).

Why do the identity tables (verify_D1_6_identity.py, verify_D1_6_swapcheck.py S3) show only d12-25 matches?
Counts matches per day group at each filter stage: any round_end row, round 1..9 complete, round 10 present
in round_state, and the final join. Also reports the round-number range and count per match for d01-07.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                for f in fl("round_end")], how="vertical_relaxed")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = RE.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
print("matches in round_end by dg:", RE.select(["match_key", "dg"]).unique().group_by("dg").len().sort("dg").to_dicts())

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "t_score", "ct_score"])
                for f in fl("round_state")], how="vertical_relaxed")
rs_mk = RS.select(["match_key", "round"]).unique()
rs_dg = rs_mk.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
rs_dg = rs_dg.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
print("matches in round_state by dg:", rs_dg.select(["match_key", "dg"]).unique().group_by("dg").len().sort("dg").to_dicts())

has10 = rs_mk.filter(pl.col("round") == 10).select("match_key").unique()
has1_9 = (rs_mk.filter(pl.col("round").is_between(1, 9)).group_by("match_key").agg(pl.col("round").n_unique().alias("n")))
ok19 = has1_9.filter(pl.col("n") == 9).select("match_key")
print("round_state matches with round 10:", has10.height, " with rounds 1..9 all present:", ok19.height)

mk_dg = RE.select(["match_key", "dg"]).unique()
for label, keys in [("round 10 in round_state", has10), ("rounds 1..9 complete", ok19),
                    ("both", has10.join(ok19, on="match_key", how="inner"))]:
    j = mk_dg.join(keys, on="match_key", how="inner")
    print(f"{label}: by dg", j.group_by("dg").len().sort("dg").to_dicts())

rounds_per_match = RE.group_by(["match_key", "dg"]).agg(pl.col("round").max().alias("max_round"), pl.len().alias("n_rounds"))
print("round_end rounds per match by dg (max round):")
print(rounds_per_match.group_by("dg").agg(pl.col("max_round").mean().alias("mean_max"),
                                          pl.col("max_round").min().alias("min_max"),
                                          pl.col("max_round").max().alias("max_max"),
                                          pl.len().alias("matches")).sort("dg"))
