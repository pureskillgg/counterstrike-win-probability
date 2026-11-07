"""Verify D7-3 (independent): per-match ledger swap-round patterns by build (reuses the ledger from verify_D7_3_d_ledger.py
via runpy; no files written). Also max round per match and how many 9961 matches show a swap after round 8 and round 12.
"""
import sys, runpy
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ns = runpy.run_path(str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_3_d_ledger.py")), run_name="ledger_ns")
LG = ns["LG"]; build = ns["build"]; RE = ns["RE"]
S = LG.filter(pl.col("swap_next") == True).group_by("match_key").agg(pl.col("round").sort().alias("swap_pairs_after_round"))
S = S.with_columns(pl.col("match_key").replace_strict(build, default=None).alias("build_num"))
S = S.with_columns(pl.col("swap_pairs_after_round").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("pattern"))
print("matches with >=1 ledger swap:", S.height, "| by build:", S.group_by("build_num").len().sort("build_num").rows())
print("swap patterns (build, pattern) top 15:")
print(S.group_by(["build_num", "pattern"]).len().sort("len", descending=True).head(15).rows())
MX = RE.group_by("match_key").agg(pl.col("round").max().alias("max_round")).with_columns(pl.col("match_key").replace_strict(build, default=None).alias("build_num"))
print("max round per match, by build (top 10 values):")
print(MX.group_by(["build_num", "max_round"]).len().sort(["build_num", "len"], descending=[False, True]).head(20).rows())
n8 = S.filter(pl.col("pattern").str.contains("^8(,|$)|,8(,|$)")).group_by("build_num").len().rows()
print("matches with a swap after round 8, by build:", n8)
n12 = S.filter(pl.col("pattern").str.contains("(^|,)12(,|$)")).group_by("build_num").len().rows()
print("matches with a swap after round 12, by build:", n12)
