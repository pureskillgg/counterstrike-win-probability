"""Verify D7-3 (independent): which source agrees with the bomb outcome, by build x round x first-after-swap flag?
Imports the ledger logic by re-running scripts/swarm2/verify_D7_3_d_ledger.py as a module namespace via exec
(own script; reads tomes only). Prints tables only.
"""
import sys, runpy
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ns = runpy.run_path(str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_3_d_ledger.py")), run_name="ledger_ns")
R = ns["R"]
print("\n== 9961, round 13: all 20-player rows by fas flag (n_total, n_with_ledger, ledger_disagree) ==")
r13_9961 = R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13))
print(r13_9961.group_by("fas_bool").agg(pl.len().alias("n_total"), pl.col("lside").is_not_null().sum().alias("n_ledger"),
      pl.col("w_dis").sum().alias("ledger_dis")).sort("fas_bool").rows())
print("== 9842, round 13: same ==")
r13_9842 = R.filter((pl.col("build_num") == 9842) & (pl.col("round") == 13))
print(r13_9842.group_by("fas_bool").agg(pl.len().alias("n_total"), pl.col("lside").is_not_null().sum().alias("n_ledger"),
      pl.col("w_dis").sum().alias("ledger_dis")).sort("fas_bool").rows())

print("\n== bomb rows: winner vs bomb and ledger vs bomb, by build x round (rounds with bomb) ==")
B = R.filter(pl.col("bomb_side").is_not_null())
B = B.with_columns(
    pl.when(pl.col("round").is_in([9, 13, 25, 27, 28])).then(pl.col("round").cast(pl.Utf8)).otherwise(pl.lit("other")).alias("rb"))
t = B.group_by(["build_num", "rb"]).agg(
    pl.len().alias("n_bomb"),
    pl.col("mm").sum().alias("winner_neq_bomb"),
    (pl.col("lside").is_not_null()).sum().alias("n_ledger"),
    ((pl.col("lside") != pl.col("bomb_side")) & pl.col("lside").is_not_null()).sum().alias("ledger_neq_bomb"),
).sort(["build_num", "rb"])
print(t)

print("\n== 9961 non-fas round-13 bomb rows: winner != bomb, ledger status ==")
x = B.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & (~pl.col("fas_bool")))
print(x.group_by(["mm", pl.col("lside").is_not_null().alias("has_ledger")]).len().rows())
print("9961 non-fas round-13 bomb rows with NO mismatch:", x.filter(~pl.col("mm")).select(["match_key", "winner_team_code", "bomb_side", "lside", "fas_bool", "fas_unknown"]).with_columns(pl.col("match_key").str.slice(0, 40)).rows())

print("\n== 9961 round-13 rows overall (20p): bomb rows vs mismatch, by fas, with unique ledger ==")
print(R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null()).group_by(["fas_bool", "mm"]).len().sort(["fas_bool", "mm"]).rows())

print("\n== 9961 round-13 rounds with winner != ledger and NO bomb outcome (count) ==")
print(R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("w_dis") & pl.col("bomb_side").is_null()).height)
print("== 9961 round-13 rounds with ledger agree and NO bomb outcome (count) ==")
print(R.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & (pl.col("w_dis") == False) & pl.col("bomb_side").is_null()).height)
