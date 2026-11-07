"""Map vs build probe for claim D1-9 (verifier run). Read-only.

Does map_name explain the 2024 build_num split (9961 vs 9842), the constant-20 player_count and the
round-13 side mismatch? Counts matches per build and map, and mismatched round-13 matches per map.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"


def read(ch, cols):
    folder = TOME24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


h = read("header", ["key", "build_num", "map_name"]).rename({"key": "match_key"})
print("== matches by build_num x map_name (top 20) ==")
print(h.group_by(["build_num", "map_name"]).len().sort("len", descending=True).head(20))
print("\n== maps per build (distinct map_name count) ==")
print(h.group_by("build_num").agg(pl.col("map_name").n_unique().alias("n_maps"), pl.len().alias("matches")))
re_ = read("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(
    ["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
out = (bs.group_by(["match_key", "round"]).agg(pl.col("event_type").n_unique().alias("n"),
                                              pl.col("event_type").first().alias("ev")).filter(pl.col("n") == 1))
j = (out.join(re_, on=["match_key", "round"]).join(h, on="match_key", how="left")
        .with_columns(pl.when(pl.col("ev") == "bomb_defused").then(3).otherwise(2).alias("bs"))
        .with_columns((pl.col("bs") != pl.col("winner_team_code")).alias("mm")))
print("\n== side mismatch by map_name x build_num (rounds with outcome, mismatches) ==")
print(j.group_by(["build_num", "map_name"]).agg(pl.len().alias("rounds"), pl.col("mm").sum().alias("mismatch"))
      .sort("mismatch", descending=True).head(12))
