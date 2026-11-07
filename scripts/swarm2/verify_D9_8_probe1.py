"""verify D9-8 probe 1: schema and event-type survey for 2024 tome (read-only).

Reads only the columns needed. Prints build distribution from header, round_state
event types and phases, and a sample of round_state score rows for one match.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


H = pl.scan_parquet(files_of("header")).select(
    ["key", "build_num", "game_directory", "map_name", "match_date", "tick_rate",
     "t_starters_score_final", "ct_starters_score_final"]).collect()
print("header rows", H.height, "unique key", H["key"].n_unique())
print("build_num dist:\n", H.group_by("build_num").len().sort("build_num"))
print("game_directory dist:\n", H.group_by("game_directory").len().sort("len", descending=True).head(10))
print("tick_rate dist:", H.group_by("tick_rate").len().rows())

RS = pl.scan_parquet(files_of("round_state")).select(
    ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"]).collect()
print("\nround_state rows", RS.height)
print("event_type:\n", RS.group_by("event_type").len().sort("len", descending=True))
print("phase:\n", RS.group_by("phase").len().sort("len", descending=True))
print("is_warmup:", RS.group_by("is_warmup").len().rows())

one = RS["match_key"][0]
print("\nsample match:", one)
print(RS.filter(pl.col("match_key") == one).filter(pl.col("round").is_in([12, 13, 14]))
      .sort(["round", "tick"]).head(60))
