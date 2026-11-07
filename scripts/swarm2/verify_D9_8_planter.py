"""verify D9-8, part 6: are player_info team codes consistent for round 13 in build 9961?
(2024 tome, read-only). A bomb can only be planted by T (team code 2). For every bomb_planted event,
look up the planter's team_code in player_info for the same (match, round). If player_info codes were
stale (pre-swap) in round 13, planters would show team_code 3.

Also tabulates the bomb_state planter team code (the e2 field) for comparison.
ASSUMPTIONS: A1 team codes 2 = T, 3 = CT. Output is printed; no files are written under E:.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


H = lz("header", ["key", "build_num"]).collect().rename({"key": "match_key"}).with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
BS = lz("bomb_state", ["match_key", "round", "event_type", "player_id", "player_team_code"]).filter(
    pl.col("event_type") == "bomb_planted").collect()
PL = BS.sort("player_id").group_by(["match_key", "round"]).agg(
    pl.col("player_id").first().alias("planter_id"),
    pl.col("player_team_code").first().alias("bs_planter_team"),
    pl.len().alias("n_plant"))
PI = lz("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
PIu = PI.unique(subset=["match_key", "round", "player_id"], keep="last").rename({"player_id": "planter_id"})
J = PL.join(PIu.select(["match_key", "round", "planter_id", "team_code"]),
            on=["match_key", "round", "planter_id"], how="left").join(
    H.select(["match_key", "build_num", "day"]), on="match_key", how="left")
J = J.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
      .when(pl.col("round") == 9).then(pl.lit("r9")).when(pl.col("round") == 28).then(pl.lit("r28"))
      .otherwise(pl.lit("other")).alias("rbin"))
print("planted (match, round) rows:", J.height, "| with a player_info row for planter:",
      int(J["team_code"].is_not_null().sum()))
print("\nplanter team_code in player_info, by build x round bin:")
print(J.group_by(["build_num", "rbin", "team_code"]).len().sort(["build_num", "rbin", "team_code"]))
print("\nplanter team_code in bomb_state (e2 field), by build x round bin:")
print(J.group_by(["build_num", "rbin", "bs_planter_team"]).len().sort(["build_num", "rbin", "bs_planter_team"]))
print("\nround 13 build 9961: planter team_code in player_info vs bomb_state (cross-tab):")
print(J.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13)).group_by(
    ["bs_planter_team", "team_code"]).len().sort(["bs_planter_team", "team_code"]))
print("\nDONE")
