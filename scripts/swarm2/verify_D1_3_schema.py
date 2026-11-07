"""verify D1-3: schema and event-type probe for the 2024 tome (read-only).

Prints column schemas and the distinct event_type values of bomb_state, and the
row count and tick range per channel for one sample file. No outputs are written.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


for ch in ["round_end", "round_start", "round_state", "bomb_state"]:
    fs = files(ch)
    print(f"== {ch}: {len(fs)} dataframe files")
    print("   schema:", pl.read_parquet_schema(fs[0]))

bs = pl.concat([pl.read_parquet(f, columns=["event_type"]) for f in files("bomb_state")])
print("\nbomb_state event_type counts:")
print(bs.group_by("event_type").len().sort("len", descending=True))

re_ = pl.concat([pl.read_parquet(f, columns=["round", "player_count"]) for f in files("round_end")])
print("\nround_end player_count distribution (all 2024 round_end rows):")
print(re_.group_by("player_count").len().sort("player_count"))
print("\nround_end total rows:", re_.height)
