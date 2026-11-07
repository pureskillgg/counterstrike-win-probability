"""D6-6 verifier step 01: is the match 'day' (key folder DD) the same as header match_date?

Also counts duplicate match identities, since day 12 has 1,004 raw match folders against about 200 on other days.
Read-only. Reads header columns only.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_01_daykey.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"

hf = sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "map_name", "match_id", "demo_id", "sharecode",
                                            "t_starters_score_final", "ct_starters_score_final", "tick_rate"])
               for f in hf], how="vertical_relaxed")
H = H.with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("key_day"),
    pl.col("match_date").str.slice(0, 10).alias("md_date"),
)
print("header rows:", H.height)
print("\n== A. header rows per key_day ==")
print(H.group_by("key_day").len().sort("key_day").transpose(include_header=True, column_names=H.group_by("key_day").len().sort("key_day")["key_day"].to_list()))
print("\n== B. key_day vs match_date (UTC date) crosstab, days with n>0 ==")
ct = H.group_by(["key_day", "md_date"]).len().sort(["key_day", "md_date"])
print(ct.group_by("key_day").agg(pl.len().alias("n_distinct_md_dates"),
                                pl.col("len").sum().alias("n"),
                                pl.col("md_date").min().alias("md_min"),
                                pl.col("md_date").max().alias("md_max")).sort("key_day"))
mis = H.filter(pl.col("key_day") != pl.col("md_date").str.slice(8, 2))
print("\nrows where key_day != match_date day-of-month:", mis.height, "of", H.height)
print(mis.group_by(["key_day", "md_date"]).len().sort("len", descending=True).head(25))

print("\n== C. duplicate identities in header ==")
for col in ["key", "match_id", "demo_id", "sharecode"]:
    print(f"{col}: rows={H.height} unique={H[col].n_unique()}")
uuid_part = H.with_columns(pl.col("key").str.extract(r"csds/2024/02/\d\d/(.+)$", 1).alias("uuid"))
print("uuid unique:", uuid_part["uuid"].n_unique(), "of", uuid_part.height)
dups = uuid_part.group_by("uuid").agg(pl.col("key_day").alias("days"), pl.len().alias("n")).filter(pl.col("n") > 1)
print("uuids appearing under more than one key_day:", dups.height)
print(dups.head(10))

print("\n== D. header per-day mean scores and tick rate (sanity) ==")
print(H.group_by("key_day").agg(pl.len().alias("n"),
                                 pl.col("tick_rate").mean().alias("tick_mean"),
                                 pl.col("map_name").n_unique().alias("n_maps")).sort("key_day"))

print("\n== E. map counts by key_day for 12 and 17 (to check map mix by day) ==")
for dday in ["01", "07", "12", "13", "17", "25"]:
    sub = H.filter(pl.col("key_day") == dday)
    print(dday, sub.group_by("map_name").len().sort("len", descending=True).head(12).rows())
