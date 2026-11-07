"""verify D1-6 (swarm2 verifier): header schema probe (2024). Read-only."""
import glob
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(40)
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
hf = sorted(glob.glob(str(T24 / "header" / "dataframe_*")))
print("header files:", hf)
print("schema:", pl.read_parquet_schema(hf[0]))
h = pl.read_parquet(hf[0])
print("rows:", h.height)
print(h.head(3))
