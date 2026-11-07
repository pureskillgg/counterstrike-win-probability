"""verify_D9_10 step 0: schema probe for the 2024-02-g tome (read-only). Prints column names and row counts only."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for name in ["player_death", "round_state", "bomb_state", "player_info", "round_end", "round_start", "player_personal", "player_disconnect"]:
    fd = BASE / f"basic_{name}{SUF}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    lf = pl.scan_parquet(files)
    sch = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"== {name}: rows={n}")
    print("   " + ", ".join(f"{k}:{v}" for k, v in sch.items()))
