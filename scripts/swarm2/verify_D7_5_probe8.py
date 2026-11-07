"""verify D7-5 step 9: physical side anchor. The planter of a bomb is always on T, so the planter's player_team_code on
bomb_planted rows should be 2 (T) in every round, in both builds, including the rounds where winner_team_code disagrees with
the bomb outcome. Also reports the round-13 disagreement count with and without the planter anchor.
Scratchpad R table (plant_team = team code of the first bomb_planted row). Read-only.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
R = pl.read_parquet(str(SCR / "R_all_rounds.parquet")).with_columns(pl.col("build_num").cast(pl.Utf8).alias("build"))
Q = R.filter(pl.col("player_count") == 20)
P = Q.filter(pl.col("plant_tick").is_not_null())
print("planted 5v5 rounds with planter team code known, by build:")
print(P.group_by(["build", "plant_team"]).len().sort(["build", "plant_team"]).rows())
B13 = Q.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null())
print("\nround-13 bomb rows by build: planter team x (winner == bomb outcome):")
print(B13.group_by(["build", "plant_team", pl.col("winner_team_code") == pl.col("bomb_side")]).len()
      .sort(["build", "plant_team"]).rows())
print("\nround-13 bomb rows (9961) with winner != bomb: planter team counts:")
print(B13.filter((pl.col("build") == "9961") & (pl.col("winner_team_code") != pl.col("bomb_side")))
      .group_by("plant_team").len().sort("plant_team").rows())
print("\nbomb outcome by planter side (all 5v5 rounds with outcome, by build):")
print(Q.filter(pl.col("bomb_side").is_not_null()).group_by(["build", "bomb_side", "plant_team"]).len()
      .sort(["build", "bomb_side", "plant_team"]).rows())
