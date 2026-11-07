"""Schema and row-count probe for disconnect/takeover parquet files used by verify_D3_3_g (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ("player_disconnect", "bot_takeover"):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    for f in fs:
        sch = pl.read_parquet_schema(f)
        n = pl.scan_parquet(f).select(pl.len()).collect().item()
        print(ch, Path(f).name, "cols:", list(sch.keys()), "rows:", n)
