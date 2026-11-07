"""D8 probe: schema, key format, and map/day coverage for the 2024 tome.

Checks that header.key == round_end.match_key, lists map_name values, and tabulates
match count by map x day group. Read-only. Uses column projection only.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(60)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read_dir(fd, cols=None):
    fs = files_of(fd)
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


hdr_file = files_of(BASE / "header")[0]
print("header schema:", dict(pl.read_parquet_schema(hdr_file)))
H = read_dir(BASE / "header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"])
print("header rows:", H.height, " unique keys:", H["key"].n_unique())
print("\nmap_name value counts (all header rows):")
print(H.group_by("map_name").len().sort("len", descending=True))

H = H.with_columns(
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
    pl.when(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int32) <= 7)
    .then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"),
)
print("\nday values:", sorted(H["day"].unique().to_list()))
print("\nmatch count by day group x map_name:")
print(H.group_by(["dg", "map_name"]).len().pivot(on="dg", index="map_name", values="len").fill_null(0).sort("map_name"))

print("\nmatch count by day x map_name (day 12-25 only):")
X = H.filter(pl.col("dg") == "d12-25").group_by(["day", "map_name"]).len().pivot(
    on="map_name", index="day", values="len").fill_null(0).sort("day")
print(X)

RE_FILE = files_of(BASE / "basic_round_end.2023-12-28,2024-02-25")[0]
print("\nround_end schema:", dict(pl.read_parquet_schema(RE_FILE)))
RE = read_dir(BASE / "basic_round_end.2023-12-28,2024-02-25", ["match_key", "round"])
print("round_end match_keys present in header:",
      RE.select(pl.col("match_key").n_unique()).item(),
      "distinct; in header:", len(set(RE["match_key"].unique().to_list()) & set(H["key"].to_list())))
