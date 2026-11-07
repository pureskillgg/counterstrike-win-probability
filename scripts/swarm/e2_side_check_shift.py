"""Test whether 2024 round-13 bomb outcomes line up with a neighbouring round's winner (index shift hypothesis).

For each 2024 clean (20-player) round with a defuse/explode row, bomb_side = 3 if defused else 2.
Compares bomb_side with round_end winner of the same round (r), the previous round (r-1) and the next (r+1),
split by round bin and by day group (days 01-07 vs 12-25).
Run: pixi run python -I -X utf8 scripts\\swarm\\e2_side_check_shift.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"]).filter(pl.col("player_count") == 20)
BS = read("bomb_state", ["match_key", "round", "event_type"])
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
bomb = outcome.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
bomb = bomb.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

prev = RE.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("winner_team_code").alias("w_prev")])
nxt = RE.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("winner_team_code").alias("w_next")])
cur = RE.select(["match_key", "round", pl.col("winner_team_code").alias("w_cur")])
j = (bomb.join(cur, on=["match_key", "round"], how="inner")
     .join(prev, on=["match_key", "round"], how="left")
     .join(nxt, on=["match_key", "round"], how="left")
     .with_columns(
         (pl.col("bomb_side") == pl.col("w_cur")).alias("eq_cur"),
         (pl.col("bomb_side") == pl.col("w_prev")).alias("eq_prev"),
         (pl.col("bomb_side") == pl.col("w_next")).alias("eq_next"),
         pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin"),
         pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dgrp"),
     ))
print("bomb rounds joined:", j.height)
print("\nround 13 only: share where bomb outcome equals each round_end winner")
r13 = j.filter(pl.col("round") == 13)
print(r13.group_by("dgrp").agg(
    pl.len().alias("n"),
    pl.col("eq_cur").mean().alias("eq_r13"),
    pl.col("eq_prev").mean().alias("eq_r12"),
    pl.col("eq_next").mean().alias("eq_r14"),
).sort("dgrp"))
print("\nother rounds (control): share equal for r, r-1, r+1")
oth = j.filter(pl.col("round") != 13)
print(oth.group_by("dgrp").agg(
    pl.len().alias("n"),
    pl.col("eq_cur").mean().alias("eq_r"),
    pl.col("eq_prev").mean().alias("eq_prev"),
    pl.col("eq_next").mean().alias("eq_next"),
).sort("dgrp"))
