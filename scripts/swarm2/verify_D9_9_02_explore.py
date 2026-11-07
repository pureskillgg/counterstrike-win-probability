"""verify_D9_9 step 2 (exploration): score-based swap detection and null-victim-team rates by build and round.

Reads only the step-1 scratchpad tables. Prints; no writes outside scratchpad.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round.parquet")

R = R.with_columns(
    (((pl.col("dT_straight") == 1) & (pl.col("dCT_straight") == 0)) |
     ((pl.col("dT_straight") == 0) & (pl.col("dCT_straight") == 1))).alias("straight_unit"),
    (((pl.col("dT_swapped") == 1) & (pl.col("dCT_swapped") == 0)) |
     ((pl.col("dT_swapped") == 0) & (pl.col("dCT_swapped") == 1))).alias("swapped_unit"),
)
print("== score-delta structure: straight unit vs swapped unit, by build (all rounds with score)")
print(R.filter(pl.col("t_after").is_not_null()).group_by(["build_num", "straight_unit", "swapped_unit"])
      .len().sort(["build_num", "straight_unit", "swapped_unit"]))

print("\n== rounds where straight delta is NOT a unit vector: by build and round index (top 40 rows)")
ns = R.filter(pl.col("t_after").is_not_null() & ~pl.col("straight_unit"))
print(ns.group_by(["build_num", "round"]).len().sort(["build_num", "round"]).filter(pl.col("len") > 0).head(80))
print("total non-straight-unit rounds:", ns.height)

print("\n== rounds where swapped delta IS a unit vector but straight is not, by build x round")
sw = R.filter(pl.col("t_after").is_not_null() & ~pl.col("straight_unit") & pl.col("swapped_unit"))
print(sw.group_by(["build_num", "round"]).len().sort(["build_num", "round"]).head(80))
print("count swapped-unit-only:", sw.height, "| non-straight & not swapped unit:",
      R.filter(pl.col("t_after").is_not_null() & ~pl.col("straight_unit") & ~pl.col("swapped_unit")).height)

print("\n== null victim team in death rows: rounds with any null victim team, by build x round class")
print(R.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("rounds"),
    (pl.col("dead_null_team") > 0).mean().round(4).alias("frac_rounds_with_null_victim"),
    pl.col("dead_null_team").sum().alias("null_victim_rows"),
    pl.col("n_dead_rows").sum().alias("death_rows")).sort(["build_num", "rclass"]))
print("\n== null victim team by build x round (rounds 8-14, 24-29)")
print(R.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29])).group_by(["build_num", "round"])
      .agg(pl.len().alias("rounds"), (pl.col("dead_null_team") > 0).mean().round(3).alias("frac_null")).sort(["build_num", "round"]))
