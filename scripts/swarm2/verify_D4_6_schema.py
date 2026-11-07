"""Schema and size probe for the verifier of D4-6. Reads parquet metadata only."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
for ch in ["round_state", "round_end", "player_death", "bomb_state"]:
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in fs)
    print(ch, "files", len(fs), "rows", n)
    print("  schema", dict(pl.read_parquet_schema(fs[0])))
