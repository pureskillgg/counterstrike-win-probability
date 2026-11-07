"""verify_D9_2_d: do player side labels flip at the expected swap rounds, per build?

Question: if the round-13 defect were stale SIDE LABELS (player_info / death / bomb team codes not updated at
the halftime swap), a player's team_code would NOT change between round 12 and round 13 in build 9961.
If the labels do update (normal), the side labels are current and the recorded round_end winner is the
suspect field.

Method:
  - player_info rows per (match_key, round, player_id), team_code (dedupe keep last).
  - For each match and each consecutive round pair (r, r+1) where the same player appears in both rounds,
    flip = team_code changed. Report flip rate per transition r, per build.
  - Expected for MR12 with OT: flips at r = 12 (->13), 24 (->25), 27 (->28); no flips elsewhere.
    ASSUMPTION: a swap is a halftime or overtime side change. Exceptions are players who join or leave.
  - Also: death victim side (player_death.player_team_code) vs player_info team_code in the same round.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    print(f"[load] {name}: missing={[c for c in cols if c not in have]}")
    return lf.select([c for c in cols if c in have]).collect()


H = scan("header", ["key", "build_num"]).with_columns(pl.col("key").alias("match_key"))
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="last")
PI = PI.join(H.select(["match_key", "build_num"]), on="match_key", how="left")

# transition index r = flip between round r and round r+1 (same player, same match)
L_ = PI.select(["match_key", "player_id", "round", "team_code", "build_num"]).rename({"team_code": "t_r"})
R_ = (PI.select(["match_key", "player_id", "round", "team_code"])
        .with_columns((pl.col("round") - 1).alias("round"))  # now round = r for the row of round r+1
        .rename({"team_code": "t_r1"}))
tr = L_.join(R_, on=["match_key", "player_id", "round"], how="inner").with_columns(
    (pl.col("t_r") != pl.col("t_r1")).alias("flip"))
print("\n== A. side flip rate, index r = flip between round r and r+1 (same player); build x r")
A = (tr.group_by(["build_num", "round"]).agg(pl.len().alias("pairs"), pl.col("flip").sum().alias("flips"))
       .with_columns((pl.col("flips") / pl.col("pairs")).round(3).alias("flip_rate"))
       .filter(pl.col("round") <= 28).sort(["build_num", "round"]))
print(A.filter(pl.col("round").is_in([2, 9, 10, 11, 12, 13, 14, 16, 22, 24, 25, 26, 27, 28, 29])))

print("\n== B. death victim side vs player_info side in the same round (match_key, round, player_id)")
DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="first")
J = DE.join(PI.select(["match_key", "round", "player_id", "team_code", "build_num"]),
            on=["match_key", "round", "player_id"], how="inner")
J = J.with_columns((pl.col("player_team_code") == pl.col("team_code")).alias("agree"))
print(J.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("agree").sum().alias("agree"),
                                              (pl.col("agree").cast(pl.Float64).mean()).round(3).alias("rate"))
        .filter(pl.col("round").is_in([9, 12, 13, 14, 25, 28])).sort(["build_num", "round"]))
print("overall death/player_info side agreement by build (n, agreeing, disagreeing):")
print(J.group_by("build_num").agg(pl.len().alias("n"), pl.col("agree").sum().alias("agreeing"),
                                  (~pl.col("agree")).sum().alias("disagreeing")).sort("build_num"))
print("\nDONE part d")
