"""Verify D7-3 (independent): header build_num per match, onset timing, and header-field change points.
Reads only the header parquet of the 2024 tome and the 2026 tome. Nothing is written except the log via stdout redirect.
ASSUMPTION: match time = header match_date parsed as UTC (as recorded, trailing Z dropped).
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
H24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds" / "header" / "dataframe_00000"
H26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds" / "header" / "dataframe_00000"

def load(path):
    h = pl.read_parquet(str(path))
    h = h.with_columns(
        pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
    return h

h = load(H24)
print("2024 header rows:", h.height, "unique key:", h["key"].n_unique(), "unique match_id:", h["match_id"].n_unique())
print("null build_num:", h["build_num"].null_count(), "null mdt:", h["mdt"].null_count())
print("build_num counts:", h.group_by("build_num").len().sort("build_num").rows())
print("demo_version_name counts:", h.group_by("demo_version_name").len().rows())
print("tick_rate counts:", h.group_by("tick_rate").len().rows())
print("map_name counts:", h.group_by("map_name").len().sort("len", descending=True).rows())

f9961 = h.filter(pl.col("build_num") == 9961)["mdt"].min()
last9842_pre = h.filter((pl.col("build_num") == 9842) & (pl.col("mdt") < f9961))["mdt"].max()
print("\nfirst build 9961 mdt:", f9961, "| last build 9842 before first 9961:", last9842_pre)
post9842 = h.filter((pl.col("build_num") == 9842) & (pl.col("mdt") > f9961)).sort("mdt")
print("build 9842 matches AFTER first 9961 (all matches, not only fas rounds):", post9842.height)
print(post9842.select(["mdt", "map_name", "key"]).with_columns(pl.col("key").str.slice(0, 40)).rows())
pre9961 = h.filter((pl.col("build_num") == 9961) & (pl.col("mdt") < pl.datetime(2024, 2, 7)))
print("build 9961 matches before 2024-02-07 00:00:", pre9961.height)
print("build 9961 by UTC day:", h.filter(pl.col("build_num") == 9961).with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("d")).group_by("d").len().sort("d").rows())
print("build 9842 by UTC day:", h.filter(pl.col("build_num") == 9842).with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("d")).group_by("d").len().sort("d").rows())

# Which header fields change exactly at the onset? compare pre (build 9842) and post (build 9961) distinct values
cols = [c for c in h.columns if c not in ("mdt", "key", "match_id", "match_date", "sharecode", "demo_id", "match_type")]
pre = h.filter(pl.col("mdt") < f9961)
post = h.filter(pl.col("mdt") >= f9961)
print("\nheader fields: n distinct pre-onset vs post-onset (rows pre={}, post={})".format(pre.height, post.height))
for c in cols:
    try:
        a = pre[c].n_unique(); b = post[c].n_unique()
        sa = set(pre[c].cast(pl.Utf8).unique().to_list()); sb = set(post[c].cast(pl.Utf8).unique().to_list())
        print(f"  {c}: pre_distinct={a} post_distinct={b} overlap={len(sa & sb)} pre_only={len(sa - sb)} post_only={len(sb - sa)}")
    except Exception as e:
        print("  ", c, "error", e)

# 2026 tome build distribution (claim said 'not done')
h26 = load(H26)
print("\n2026 header rows:", h26.height, "unique key:", h26["key"].n_unique())
print("2026 build_num counts:", h26.group_by("build_num").len().sort("build_num").rows())
print("2026 demo_version_name counts:", h26.group_by("demo_version_name").len().rows())
print("2026 mdt range:", h26["mdt"].min(), h26["mdt"].max())
print("2026 match_date sample:", h26["match_date"].head(3).to_list())
