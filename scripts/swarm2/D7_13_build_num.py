"""D7-13: does the demo header build_num (and client/server name fields) change at the onset window?
Time = header match_date (UTC). Onset window from D7-11: last agreeing 2024-02-07 00:24:22, first flip 01:01:18 UTC.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "demo_version_name", "server_name"]) \
    .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
print("matches:", H.height, "| distinct build_num:", H["build_num"].n_unique(), "| distinct demo_version_name:", H["demo_version_name"].n_unique())
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
S = H.sort("mdt")
# step points: first match with each build_num, and build_num value just before/after the onset window
w = S.filter((pl.col("mdt") >= pl.datetime(2024, 2, 6, 12)) & (pl.col("mdt") < pl.datetime(2024, 2, 8, 12)))
print("\nbuild_num by match_date hour, 2024-02-06 12:00 .. 2024-02-08 12:00 (UTC):")
print(w.with_columns(pl.col("mdt").dt.strftime("%m-%d %H").alias("hr")).group_by(["hr", "build_num"]).len().sort("hr").rows())
pre = S.filter(pl.col("mdt") < pl.datetime(2024, 2, 7, 0, 24, 22))["build_num"].unique().to_list()
post = S.filter(pl.col("mdt") >= pl.datetime(2024, 2, 7, 1, 1, 18))["build_num"].unique().to_list()
print("\nbuild_num before 00:24:22 Feb 7:", sorted(pre))
print("build_num from 01:01:18 Feb 7 onward:", sorted(post))
print("build_num values present on both sides:", sorted(set(pre) & set(post)))
