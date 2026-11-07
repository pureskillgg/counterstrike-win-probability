"""verify D7-2, step 1: probe raw header fields for the 2024 tome.
Checks: match_date string formats (is the microsecond timestamp unusual?), raw strings for the two matches named in the claim,
path day (csds/2024/MM/DD) versus header match_date day, and build_num versus path day.
Memory: header file only (about 4,435 rows, few columns).
"""
import sys, re, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
hfiles = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("header files:", len(hfiles))
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name", "match_id"]) for f in hfiles], how="vertical_relaxed")
H = H.rename({"key": "match_key"})
print("header rows:", H.height, "| distinct match_key:", H["match_key"].n_unique())

# raw match_date format classes
def fmt_class(s):
    if s is None:
        return "NULL"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.000Z", s):
        return "ISO_ms000_Z"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s):
        return "ISO_noframe_Z"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", s):
        return "ISO_fraction_Z_nonzero_or_other"
    return "OTHER:" + s[:40]

H = H.with_columns(pl.col("match_date").map_elements(fmt_class, return_dtype=pl.Utf8).alias("mdfmt"))
print("\nmatch_date raw format classes:")
print(H.group_by("mdfmt").len().sort("len", descending=True).head(20))

odd = H.filter(~pl.col("mdfmt").eq("ISO_ms000_Z"))
print("\nnon-standard match_date rows:", odd.height)
print(odd.select(["match_key", "match_date", "mdfmt", "build_num"]).head(30))

# the two matches named in the claim, by key prefix
for pref in ["csds/2024/02/12/78a91cb9", "csds/2024/02/12/9cfb21ee", "csds/2024/02/21/2e483e20"]:
    print("\nprefix", pref, "->", H.filter(pl.col("match_key").str.starts_with(pref)).select(["match_key", "match_date", "build_num", "match_id"]).rows())

# path day vs header day
H = H.with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("path_day"),
    pl.col("match_date").str.slice(0, 10).str.slice(8, 2).alias("md_day_raw"),
    pl.col("match_date").str.slice(0, 7).alias("md_month_raw"),
)
print("\npath_day null:", H.filter(pl.col("path_day").is_null()).height,
      "| path_day values:", sorted(H["path_day"].drop_nulls().unique().to_list()))
print("\nmatch_date month values:", H["md_month_raw"].value_counts().sort("md_month_raw").rows())
print("\nbuild_num x path_day:")
print(H.group_by(["path_day", "build_num"]).len().sort(["path_day", "build_num"]).rows())
print("\nbuild_num x md_day_raw (match_date day, UTC string):")
print(H.group_by(["md_day_raw", "build_num"]).len().sort(["md_day_raw", "build_num"]).rows())

# path day versus match_date day disagreement
dis = H.filter(pl.col("path_day") != pl.col("md_day_raw"))
print("\nmatches whose path day != match_date day:", dis.height, "of", H.height)
print(dis.group_by(["path_day", "md_day_raw"]).len().sort(["path_day", "md_day_raw"]).rows())
