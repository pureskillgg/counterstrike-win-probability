"""2024 round-13 side mismatch: which source is right?

score_update is not in the 2024 tome, so the check uses the planter side: a bomb can only be planted by T (code 2).
For each 2024 round with a bomb_planted row and a defuse/explode row, compares:
  - winner_team_code (round_end) vs bomb outcome (defused -> 3, exploded -> 2)
  - planter player_team_code on bomb_planted rows (should be 2 = T)
Splits by round number bin and by match date to see if the mismatch is a tome/date artifact.
Run: pixi run python -I -X utf8 scripts\\swarm\\e2_side_check_r13.py
"""
import glob
import re
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(fd)], how="vertical_relaxed")


RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
print("round_end columns:", list(pl.read_parquet_schema(files_of(BASE / f'basic_round_end{SUF}')[0]).keys()))
BS = read("bomb_state", ["match_key", "round", "event_type", "player_team_code"])

outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
planter = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("player_team_code").first().alias("planter_side")))

j = (RE.filter(pl.col("player_count") == 20)
     .join(outcome, on=["match_key", "round"], how="inner")
     .join(planter, on=["match_key", "round"], how="left")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
         pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
     )
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                   pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
                   .otherwise(pl.lit("other")).alias("rbin")))

print(f"\nrounds with outcome: {j.height}, mismatches: {int(j['mismatch'].sum())}")
print("\nplanter side (should be 2 = T) for planted rounds, by mismatch flag and round bin:")
print(j.group_by(["rbin", "mismatch", "planter_side"]).len().sort(["rbin", "mismatch", "planter_side"]))

print("\nmismatch rate by round bin:")
print(j.group_by("rbin").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                             (pl.col("mismatch").mean()).alias("rate")).sort("rbin"))

print("\nround-13 mismatch rate by match day:")
r13 = j.filter(pl.col("round") == 13)
print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                              pl.col("mismatch").mean().alias("rate")).sort("day"))

print("\nround-13 winner_team_code counts (all clean round-13 rows, incl. no bomb outcome):")
r13_all = RE.filter((pl.col("player_count") == 20) & (pl.col("round") == 13))
print(r13_all.group_by("winner_team_code").len().sort("winner_team_code"))
