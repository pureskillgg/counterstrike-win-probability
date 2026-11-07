"""Schema probe for verify_D3_3: column names and dtypes for 2024 tome channels (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ["header", "round_state", "round_end", "bomb_state", "round_freeze", "player_state", "player_round", "round_start"]:
    d = TOME / (ch if ch == "header" else f"basic_{ch}{SUF}")
    if not d.exists():
        print(ch, "MISSING", d)
        continue
    files = sorted(glob.glob(str(d / "dataframe_*")))
    sch = pl.read_parquet_schema(files[0])
    print(f"== {ch}: files={len(files)}")
    for k, v in sch.items():
        print(f"   {k}: {v}")
