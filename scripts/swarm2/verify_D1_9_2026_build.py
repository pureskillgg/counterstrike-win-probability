"""2026 build check for claim D1-9 (verifier run). Read-only.

Does the 2026 tome show the same build-dependent player_count pattern seen in 2024 build 9961?
Per build_num: share of rounds with round_end.player_count == distinct players in player_info,
and the share with player_count == 10 and a 5 v 5 player_info structure. Also the 2026 side
mismatch (bomb outcome vs winner) by round 13 and other rounds, per build.
"""
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF26 = ".2026-09-18,2026-10-08"
T, CT = 2, 3


def files(ch):
    folder = TOME26 / ("header" if ch == "header" else f"basic_{ch}{SUF26}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no files for {ch}")
    return fs


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


hdr = read("header", ["key", "build_num"]).rename({"key": "match_key"})
re_ = read("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
pi = read("player_info", ["match_key", "round", "team_code", "player_id"])
pr = pi.group_by(["match_key", "round"]).agg(
    pl.col("player_id").n_unique().alias("nd"),
    pl.col("player_id").filter(pl.col("team_code") == T).n_unique().alias("nT"),
    pl.col("player_id").filter(pl.col("team_code") == CT).n_unique().alias("nCT"))
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
out = (bs.group_by(["match_key", "round"]).agg(
           pl.col("event_type").n_unique().alias("n_types"),
           pl.col("event_type").first().alias("ev"))
       .filter(pl.col("n_types") == 1))
R = (re_.join(pr, on=["match_key", "round"], how="left")
        .join(hdr, on="match_key", how="left")
        .with_columns(
            (pl.col("player_count") == pl.col("nd")).alias("pc_eq_nd"),
            ((pl.col("player_count") == 10) & (pl.col("nT") == 5) & (pl.col("nCT") == 5)).alias("clean5v5"),
            pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("grp")))
print("== 2026 rounds by build: pc == distinct players, clean 5v5 structure, side mismatch ==")
g = R.group_by("build_num").agg(
    pl.len().alias("rounds"),
    pl.col("pc_eq_nd").mean().alias("share_pc_eq_nd"),
    pl.col("clean5v5").mean().alias("share_clean5v5"),
    pl.col("player_count").eq(10).mean().alias("share_pc10"))
print(g.sort("build_num"))
bo = out.join(re_, on=["match_key", "round"], how="inner").join(hdr, on="match_key", how="left").with_columns(
    pl.when(pl.col("ev") == "bomb_defused").then(3).otherwise(2).alias("bomb_side"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("grp"))
bo = bo.with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
print("\n== 2026 side mismatch by build and group ==")
print(bo.group_by(["build_num", "grp"]).agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mismatches")).sort(["build_num", "grp"]))
print(f"\n2026 total rounds with a single bomb outcome type: {bo.height:,}; mismatches: {int(bo['mismatch'].sum())}")
