"""verify D9-10 (h): is the 8-round-halftime format tied to map_name? Read-only, lazy scans.

For build 9961 and 9842: halftime-marker pattern per match (round_state phase == 'halftime' rounds) vs map_name.
Shows counts and within-map shares, so the log can say whether map_name is a usable proxy for the format.
The inversion rule itself uses the halftime marker, not the map name.
ASSUMPTION: the marker pattern '8' = 8-round halves; '12' = standard MR12; '12,24,27' = MR12 plus overtime.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"[{name}] missing columns {missing}")
    return lf.select(cols).collect()


H = scan("header", ["key", "build_num", "map_name"]).rename({"key": "match_key"})
H = H.filter(pl.col("build_num").is_in([9961, 9842]))
keys = H["match_key"].to_list()

RS = scan("round_state", ["match_key", "round", "phase"])
ht = (RS.filter(pl.col("phase") == "halftime").select(["match_key", "round"]).unique()
      .filter(pl.col("match_key").is_in(keys)))
ht_by = ht.sort("round").group_by("match_key", maintain_order=True).agg(pl.col("round").alias("hr"))
ht_by = ht_by.with_columns(pl.col("hr").list.eval(pl.element().cast(pl.String)).list.join(",").alias("pat"))
M = H.join(ht_by.select(["match_key", "pat"]), on="match_key", how="left").with_columns(
    pl.col("pat").fill_null("none"))
M = M.with_columns(pl.when(pl.col("pat") == "8").then(pl.lit("halftime_8"))
                   .when(pl.col("pat").str.starts_with("12,24")).then(pl.lit("halftime_12_OT"))
                   .when(pl.col("pat") == "12").then(pl.lit("halftime_12"))
                   .otherwise(pl.lit("none_or_other")).alias("fmt"))

for bld in [9961, 9842]:
    sub = M.filter(pl.col("build_num") == bld)
    print(f"== build {bld}: matches {sub.height}")
    print(sub.group_by("fmt").len().sort("fmt").rows())
    tot = sub.group_by("map_name").agg(pl.len().alias("n_all"))
    h8 = sub.filter(pl.col("fmt") == "halftime_8").group_by("map_name").agg(pl.len().alias("n_halftime8"))
    t = tot.join(h8, on="map_name", how="left").with_columns(pl.col("n_halftime8").fill_null(0))
    t = t.with_columns((pl.col("n_halftime8") / pl.col("n_all")).round(3).alias("share_halftime8"))
    print("  per map: n_all, n_halftime8, share (top 12 by n_all):")
    print(t.sort("n_all", descending=True).head(12).rows())
    print("  maps with any halftime_8 match:", t.filter(pl.col("n_halftime8") > 0).height,
          "| maps with zero:", t.filter(pl.col("n_halftime8") == 0).height)
    print()
