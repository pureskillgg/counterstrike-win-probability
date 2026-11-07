"""verify D6-6, step 01: 2024 header build/version fields by match day (header is one small parquet)."""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
HDR = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\header\dataframe_00000")
H = pl.read_parquet(str(HDR), columns=["key", "match_date", "map_name", "build_num", "demo_version_name",
                                        "demo_version_guid", "tick_rate", "server_name", "platform", "match_type"])
H = H.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("rows:", H.height, "unique keys:", H["key"].n_unique(), "day null:", H["day"].null_count())
print("day counts:", dict(sorted(H.group_by("day").len().iter_rows())))
print("match_date sample:", H["match_date"].head(5).to_list())
print("\n-- distinct build_num / demo_version_name / guid / tick_rate / server / platform / match_type")
for c in ["build_num", "demo_version_name", "demo_version_guid", "tick_rate", "platform", "match_type"]:
    print(f"{c}: n_distinct={H[c].n_unique()} top={H.group_by(c).len().sort('len', descending=True).head(5).rows()}")
print("\n-- build_num by day (matches)")
print(H.group_by(["day", "build_num"]).len().sort(["day", "build_num"]))
