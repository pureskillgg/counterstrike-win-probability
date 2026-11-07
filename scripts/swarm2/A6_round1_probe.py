"""A6 probe: round 1 coverage in 2024 round_end, by match day and by match version.

Q1. For every 2024 match (raw, not deduplicated): has round_end round 1? has round_start round 1?
    has round_state round_freeze_end round 1? has round_state round_end round 1?
Q2. Round-1 presence by match day group (01-07 vs 12-25) and by version proxy (pure elimination codes).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_round1_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def load(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*")))).select(cols).collect()


H = load("header", ["key"]) if False else None
hdr = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(["key", "match_id"]).collect()
RE = load("round_end", ["match_key", "round", "win_reason_code"])
ms = pl.DataFrame({"match_key": sorted(RE["match_key"].unique().to_list())})
ms = ms.with_columns(pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"))
r1_re = set(RE.filter(pl.col("round") == 1)["match_key"].to_list())
RST = load("round_start", ["match_key", "round"])
r1_rst = set(RST.filter(pl.col("round") == 1)["match_key"].to_list())
RS = load("round_state", ["match_key", "round", "event_type"]).filter(pl.col("round") == 1)
r1_fe = set(RS.filter(pl.col("event_type") == "round_freeze_end")["match_key"].to_list())
r1_rend = set(RS.filter(pl.col("event_type") == "round_end")["match_key"].to_list())
r1_any_rs = set(RS["match_key"].to_list())
ms = ms.with_columns(
    pl.col("match_key").is_in(list(r1_re)).alias("re_r1"),
    pl.col("match_key").is_in(list(r1_rst)).alias("rst_r1"),
    pl.col("match_key").is_in(list(r1_fe)).alias("fe_r1"),
    pl.col("match_key").is_in(list(r1_rend)).alias("rend_r1"),
    pl.col("match_key").is_in(list(r1_any_rs)).alias("any_rs_r1"))
print("== 2024 raw matches:", ms.height)
print("round_end r1 present:", ms.group_by("re_r1").len().rows())
print("round_start r1 present:", ms.group_by("rst_r1").len().rows())
print("round_state freeze_end r1 present:", ms.group_by("fe_r1").len().rows())
print("round_state round_end r1 present:", ms.group_by("rend_r1").len().rows())
print("round_state any row for round 1:", ms.group_by("any_rs_r1").len().rows())
print("\n== round_end r1 by day")
print(ms.group_by("day").agg(pl.len().alias("matches"), pl.col("re_r1").sum().alias("with_re_r1"),
                              pl.col("fe_r1").sum().alias("with_fe_r1")).sort("day"))
# min round in round_end per match for missing-r1 matches, and number of rounds
mr = RE.group_by("match_key").agg(pl.col("round").min().alias("mn"), pl.len().alias("n"))
print("\n== min round / rows for matches without round_end r1")
print(mr.filter(pl.col("match_key").is_in(list(set(ms["match_key"].to_list()) - r1_re))).group_by(["mn", "n"]).len().sort(
    "len", descending=True).head(8))
