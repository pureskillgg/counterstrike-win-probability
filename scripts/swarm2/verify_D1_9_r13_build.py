"""Side-mismatch by build for claim D1-9 context (verifier run). Read-only.

Recomputes the bomb-outcome vs winner_team_code mismatch on the 2024 tome, split by build_num and
by round 13 versus other rounds. Bomb side rule as in scripts/swarm2/D1_base_check.py:
bomb_defused -> CT (3); bomb_exploded -> T (2). Only rounds with exactly one outcome are counted.
Purpose: check whether the open round-13 side defect is a build-dependent effect, which would
make build_num a confounder for the player_count finding.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"


def read(ch, cols):
    folder = TOME24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


hdr = read("header", ["key", "build_num", "match_date"]).rename({"key": "match_key"})
re_ = read("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
out = (bs.group_by(["match_key", "round"]).agg(
           pl.col("event_type").n_unique().alias("n_types"),
           pl.col("event_type").first().alias("ev"),
           pl.len().alias("n_ev"))
       .filter(pl.col("n_types") == 1))
j = (out.join(re_, on=["match_key", "round"], how="inner")
        .join(hdr, on="match_key", how="left")
        .with_columns(
            pl.when(pl.col("ev") == "bomb_defused").then(3).otherwise(2).alias("bomb_side"))
        .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                      pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("grp"),
                      pl.col("match_date").str.slice(0, 10).alias("date")))
print(f"rounds with one bomb outcome type joined to round_end: {j.height:,}")
print(j.group_by(["build_num", "grp"]).agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mismatches"),
                                          (pl.col("mismatch").mean()).alias("rate")).sort(["build_num", "grp"]))
print("\nmismatch rate by build and player_count (rounds with outcome):")
print(j.group_by(["build_num", "player_count"]).agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mismatches"),
                                                   pl.col("mismatch").mean().alias("rate")).sort(["build_num", "rounds"], descending=[False, True]).head(12))
print("\nround 13 mismatch by date (top dates):")
print(j.filter(pl.col("round") == 13).group_by("date").agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mismatches"),
                                                          pl.col("build_num").n_unique().alias("n_builds")).sort("date"))
