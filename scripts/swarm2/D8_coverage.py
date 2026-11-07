"""D8 coverage: how many round-13 rows can the bomb check evaluate, by map?

The bomb-outcome check only applies to planted rounds. This reports, by map and day group, the round-13 rows
(player_count == 20, deduplicated) and the planted share, so a map-level selection effect can be ruled in or out.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_coverage.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
OUTCOMES = ["bomb_defused", "bomb_exploded"]


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read_dir(fd, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(fd)], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(BASE / f"basic_{ch}{SUF}", cols)


H = read_dir(BASE / "header", ["key", "map_name"])
RE = read("round_end", ["match_key", "round", "player_count"]).filter(pl.col("player_count") == 20)
RE = RE.unique(["match_key", "round"], keep="first").filter(pl.col("round") == 13)
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
planted = bs.select(["match_key", "round"]).unique()
C = (RE.join(planted.with_columns(pl.lit(True).alias("planted")), on=["match_key", "round"], how="left")
     .with_columns(pl.col("planted").fill_null(False))
     .join(H.rename({"key": "match_key"}), on="match_key", how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int32).alias("day"))
     .with_columns(pl.when(pl.col("day") <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg")))
print(f"round-13 rows (player_count==20, dedup): {C.height}; planted: {int(C['planted'].sum())}")
tab = (C.group_by(["dg", "map_name"]).agg(pl.len().alias("r13_rows"), pl.col("planted").sum().alias("planted"))
       .with_columns((pl.col("planted") / pl.col("r13_rows")).alias("planted_share")).sort(["dg", "map_name"]))
print(tab)
c2 = C.filter(pl.col("day") >= 12)
print("\ndays 12-25 totals by map:")
print(c2.group_by("map_name").agg(pl.len().alias("r13_rows"), pl.col("planted").sum().alias("planted"),
                                  pl.col("planted").mean().alias("planted_share")).sort("map_name"))
print("\nDONE")
