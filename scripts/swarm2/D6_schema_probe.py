import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
H24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds" / "header" / "dataframe_00000"
H26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds" / "header"
print("2024 header schema:", pl.read_parquet_schema(H24))
import glob
f26 = sorted(glob.glob(str(H26 / "dataframe_*")))
print("2026 header files:", len(f26))
print("2026 header schema:", pl.read_parquet_schema(f26[0]))
df = pl.read_parquet(H24)
print("2024 header rows:", df.height)
print(df.head(3).to_dicts())
