"""verify D1-3: schema probe for player_death / player_info / player_spawn (read-only)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ["player_death", "player_info", "player_spawn", "round_start"]:
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    print(f"== {ch}: {len(fs)} files")
    print(pl.read_parquet_schema(fs[0]))
    df = pl.read_parquet(fs[0]).head(3)
    print(df)
