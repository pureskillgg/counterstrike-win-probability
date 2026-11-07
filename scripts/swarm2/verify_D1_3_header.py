"""verify D1-3: header schema and final-result columns (read-only, 2024 tome)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("header files:", len(hf))
print(pl.read_parquet_schema(hf[0]))
H = pl.read_parquet(hf[0]).head(2)
print(H.transpose(include_header=True))
