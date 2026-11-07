"""M3 probe: header schema and map_name values in the 2026 tome (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
HDR = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds" / "header"
files = sorted(glob.glob(str(HDR / "dataframe_*")))
print("header pages:", len(files))
lf = pl.concat([pl.scan_parquet(f) for f in files], how="vertical_relaxed")
print("schema:")
for k, v in lf.collect_schema().items():
    print(f"  {k}: {v}")
df = lf.select(["key", "map_name", "match_date", "max_rounds"]).collect()
print("rows:", df.height, "unique keys:", df["key"].n_unique())
print("map_name null count:", df["map_name"].null_count())
vc = df.group_by("map_name").agg(pl.len().alias("header_rows")).sort("header_rows", descending=True)
print(vc)
print(df.head(3))
