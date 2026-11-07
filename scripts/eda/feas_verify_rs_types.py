"""Scratch probe (read-only): round_state event types and their timing relative to round_end."""
import glob
from collections import Counter
from pathlib import Path
import polars as pl

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
folder = [p for p in TOME.iterdir() if p.is_dir() and p.name.startswith("basic_round_state.")][0]
rs = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type"])
                for f in sorted(glob.glob(str(folder / "dataframe_*")))])
print("round_state event types:", sorted(Counter(rs["event_type"].to_list()).items()))
