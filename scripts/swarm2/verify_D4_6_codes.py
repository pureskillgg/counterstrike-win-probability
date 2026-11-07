"""Check player_death team codes and whether the 2/3 filter drops deaths; test death completeness by gap regime.

  C1. player_death.player_team_code value counts (all 2024 deaths), and for deaths inside the live window.
  C2. For full rounds with win reason 8 (CTs_Win => all 5 T dead) or 9 (Terrorists_Win => all 5 CT dead):
      count deaths of each team code in the live window, including codes other than 2/3.
  C3. Death completeness by gap regime (0 vs 429 vs other): P(5 T deaths) for reason 8 rounds, P(5 CT deaths) for reason 9.
ASSUMPTIONS: A9. A full round in 2024 has 5 players per side (player_count 20 = 2 rows per player in round_end; 2026 has 10).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
import glob as _g


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(_g.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


de = read("player_death", ["match_key", "round", "tick", "player_team_code", "player_id", "attacker_id"])
print("C1 player_team_code value counts (all deaths):")
print(de.group_by("player_team_code", maintain_order=False).len().sort("len", descending=True))
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
FULL = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & pl.col("live").is_not_null() & pl.col("end").is_not_null())
w = de.join(FULL.select(["match_key", "round", "live", "end", "win_reason_code", "S", "W", "gap"]), on=["match_key", "round"], how="inner")
w = w.filter((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
agg = w.group_by(["match_key", "round"]).agg(
    (pl.col("player_team_code") == 2).sum().alias("nT2"),
    (pl.col("player_team_code") == 3).sum().alias("nCT3"),
    (~pl.col("player_team_code").is_in([2, 3])).sum().alias("n_other_code"),
    pl.col("player_team_code").is_null().sum().alias("n_null_code"),
    pl.col("player_id").n_unique().alias("n_unique_victims"),
    pl.len().alias("n_all_deaths"),
)
J = FULL.join(agg, on=["match_key", "round"], how="left")
print("\nC2: rounds with other/null team codes in window (days 12-25 full rounds):",
      J.filter((pl.col("n_other_code") > 0) | (pl.col("n_null_code") > 0)).height, "of", J.height)
print("C2: rounds where a death is recorded twice for the same player_id in the window:",
      J.filter(pl.col("n_unique_victims") < pl.col("n_all_deaths")).height)
K = J.filter((pl.col("win_reason_code") == 8) & (pl.col("S") != 0)).with_columns(
    pl.when(pl.col("gap") == 0).then(pl.lit("gap0"))
    .when(pl.col("gap") == 429).then(pl.lit("gap429")).otherwise(pl.lit("gap_other")).alias("regime"),
    pl.when(pl.col("S") != pl.col("W")).then(pl.lit("flag")).otherwise(pl.lit("agree")).alias("status"))
print("\nC3 reason 8 (CTs_Win; expect 5 T deaths): P(nT2==5) by regime x status, with CT deaths")
print(K.group_by(["status", "regime"]).agg(
    pl.len().alias("n"),
    (pl.col("nT2") == 5).mean().alias("P_5T"),
    (pl.col("nCT3") == 5).mean().alias("P_5CT"),
    pl.col("nT2").mean().alias("mean_T"),
    pl.col("nCT3").mean().alias("mean_CT"),
    pl.col("n_other_code").sum().alias("n_other_code_sum"),
).sort(["status", "regime"]))
K9 = J.filter((pl.col("win_reason_code") == 9) & (pl.col("S") != 0)).with_columns(
    pl.when(pl.col("gap") == 0).then(pl.lit("gap0"))
    .when(pl.col("gap") == 429).then(pl.lit("gap429")).otherwise(pl.lit("gap_other")).alias("regime"),
    pl.when(pl.col("S") != pl.col("W")).then(pl.lit("flag")).otherwise(pl.lit("agree")).alias("status"))
print("\nC3 reason 9 (Terrorists_Win; expect 5 CT deaths): P(nCT3==5) by regime x status")
print(K9.group_by(["status", "regime"]).agg(
    pl.len().alias("n"),
    (pl.col("nCT3") == 5).mean().alias("P_5CT"),
    (pl.col("nT2") == 5).mean().alias("P_5T"),
    pl.col("nCT3").mean().alias("mean_CT"),
    pl.col("nT2").mean().alias("mean_T"),
).sort(["status", "regime"]))
