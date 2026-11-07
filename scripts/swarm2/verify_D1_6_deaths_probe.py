"""verify D1-6 deaths probe (swarm2 verifier): schemas for player_death and player_info (2024 tome)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ["player_death", "player_info"]:
    files = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    lf = pl.scan_parquet(files)
    print(ch, "files:", len(files))
    print(lf.collect_schema())
    print("rows:", lf.select(pl.len()).collect().item())
    print(lf.head(3).collect())
