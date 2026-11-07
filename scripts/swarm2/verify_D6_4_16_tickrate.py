"""D6-4 verifier, step 16: tick_rate and match_date from the 2024 header (projection only)."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
base = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\header")
h = pl.scan_parquet(str(base / "dataframe_*")).select(["tick_rate", "tick_save_rate", "match_date", "map_name"]).collect()
print("tick_rate counts:", h.group_by("tick_rate").len().sort("tick_rate").rows())
print("tick_save_rate counts:", h.group_by("tick_save_rate").len().sort("tick_save_rate").rows())
print("map_name counts (top 5):", h.group_by("map_name").len().sort("len", descending=True).head(5).rows())
