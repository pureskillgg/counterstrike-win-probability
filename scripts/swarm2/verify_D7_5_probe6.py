"""verify D7-5 step 7: are there non-swap defects? Rounds with no bomb outcome and no elimination, split by plant status.

Physical rules used (ASSUMPTIONS, standard CS rules):
  - a round with no bomb plant and no elimination ends on time -> CT (3) wins
  - T (2) can win without a bomb outcome only by eliminating CT (elim) or via a planted-round outcome row that is missing
Reports winner_team_code and dp_side distributions for these rounds by build, with and without a plant event.
Scratchpad R table only (plant_tick, bomb_side, elim_side, winner_team_code, dp_side, round).
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D7_5")
R = pl.read_parquet(str(SCR / "R_all_rounds.parquet")).with_columns(
    pl.col("build_num").cast(pl.Utf8).alias("build"))
Q = R.filter((pl.col("player_count") == 20) & pl.col("winner_team_code").is_not_null())
NE = Q.filter(pl.col("bomb_side").is_null() & pl.col("elim_side").is_null()).with_columns(
    pl.when(pl.col("plant_tick").is_not_null()).then(pl.lit("planted_no_outcome"))
      .otherwise(pl.lit("no_plant")).alias("plant_status"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round") == 25).then(pl.lit("r25"))
      .when(pl.col("round") == 9).then(pl.lit("r09"))
      .when(pl.col("round") == 27).then(pl.lit("r27"))
      .otherwise(pl.lit("other")).alias("cls"))
print("== rounds with no bomb outcome and no elimination (5v5), by build x plant status ==")
print(NE.group_by(["build", "plant_status"]).agg(
    pl.len().alias("n"),
    (pl.col("winner_team_code") == 2).sum().alias("winner_T"),
    (pl.col("dp_side") == 2).sum().alias("dp_T"),
    pl.col("dp_side").is_not_null().sum().alias("n_dp"),
    pl.col("nT").mean().alias("mean_nT"),
    pl.col("nCT").mean().alias("mean_nCT"),
).sort(["build", "plant_status"]))
print("\n== no_plant rounds by class and build: winner_T / n ==")
print(NE.filter(pl.col("plant_status") == "no_plant").group_by(["build", "cls"]).agg(
    pl.len().alias("n"), (pl.col("winner_team_code") == 2).sum().alias("winner_T"),
    (pl.col("dp_side") == 2).sum().alias("dp_T")).sort(["build", "cls"]))
print("\n== planted_no_outcome rounds by class and build: winner_T / n ==")
print(NE.filter(pl.col("plant_status") == "planted_no_outcome").group_by(["build", "cls"]).agg(
    pl.len().alias("n"), (pl.col("winner_team_code") == 2).sum().alias("winner_T"),
    (pl.col("dp_side") == 2).sum().alias("dp_T")).sort(["build", "cls"]))
print("\nplant events per 5v5 round (planted rounds share, incl. outcome rounds), by build:")
print(Q.group_by("build").agg(pl.len().alias("rounds"), pl.col("plant_tick").is_not_null().sum().alias("planted"),
                              pl.col("bomb_side").is_not_null().sum().alias("with_outcome")).sort("build"))
