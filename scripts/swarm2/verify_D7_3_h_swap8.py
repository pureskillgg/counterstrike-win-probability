"""Verify D7-3 (independent): for matches with a ledger swap after round 8, what player counts and builds do they have,
and does the round-9 winner agree with the ledger? Reuses the ledger via runpy. No files written.
"""
import sys, runpy
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ns = runpy.run_path(str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_3_d_ledger.py")), run_name="ledger_ns")
LG = ns["LG"]; build = ns["build"]; RE = ns["RE"]
s8 = LG.filter((pl.col("swap_next") == True) & (pl.col("round") == 8)).select("match_key").unique()
print("matches with swap after round 8:", s8.height)
s8 = s8.with_columns(pl.col("match_key").replace_strict(build, default=None).alias("build_num"))
pc = RE.join(s8.select("match_key"), on="match_key", how="inner")
print("player_count distribution on rounds of swap-after-8 matches, by build (top):")
print(pc.join(s8, on="match_key").group_by(["build_num", "player_count"]).len().sort(["build_num", "len"], descending=[False, True]).head(20).rows())
r9 = pc.filter(pl.col("round") == 9).join(s8, on="match_key")
print("round 9 rows (any player count) in swap-after-8 matches, by build x player_count:")
print(r9.group_by(["build_num", "player_count"]).len().sort(["build_num", "player_count"]).rows())
mx = RE.join(s8.select("match_key"), on="match_key").group_by("match_key").agg(pl.col("round").max().alias("max_round")).join(s8, on="match_key")
print("max round in swap-after-8 matches, by build (count per max_round, top):")
print(mx.group_by(["build_num", "max_round"]).len().sort(["build_num", "len"], descending=[False, True]).head(12).rows())
print("swap-after-8 matches where round 8 itself is 20p:", RE.filter((pl.col("round") == 8) & (pl.col("player_count") == 20)).join(s8, on="match_key").height)
