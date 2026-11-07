"""Verifier probe 2: match_key format, round_end duplication, round_state score fields."""
import glob
import os
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(90)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"


def scan(ch):
    fd = os.path.join(BASE, f"basic_{ch}{SUF}")
    return pl.scan_parquet(sorted(glob.glob(os.path.join(fd, "dataframe_*"))), missing_columns="insert",
                           extra_columns="ignore")


re_ = scan("round_end").collect()
print("round_end rows", re_.height, "matches", re_["match_key"].n_unique())
print(re_.select("match_key").head(3))
dup = re_.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"),
                                              pl.col("tick").n_unique().alias("nt"))
print("rows per (match,round):")
print(dup.group_by("n").agg(pl.len().alias("pairs")).sort("n"))
print("pairs with conflicting winners:", dup.filter(pl.col("nw") > 1).height)
print("rounds per match (distinct round numbers):")
print(dup.group_by("match_key").agg(pl.len().alias("r")).group_by("r").agg(pl.len().alias("matches")).sort("r").head(40))
print("round number range:", re_["round"].min(), re_["round"].max())
rs = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"]).collect()
print("freeze_end rows", rs.height)
print("t_score != t_score_raw:", rs.filter(pl.col("t_score") != pl.col("t_score_raw")).height,
      " ct differs:", rs.filter(pl.col("ct_score") != pl.col("ct_score_raw")).height)
print(rs.select(pl.col("round").min(), pl.col("round").max()))
print(rs.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("pairs")))
