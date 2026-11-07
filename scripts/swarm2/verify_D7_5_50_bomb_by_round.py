"""verify_D7_5_50_bomb_by_round: bomb-anchored check per round number and period.

Bomb outcome (defused -> CT, exploded -> T) is the only label here that does not depend on team identity,
orientation, the score ledger, or round_end. For each (period, round) it counts how often each label agrees
with the bomb outcome, and for rows where code and within-round disagree, which one the bomb supports.
Also checks the header fields for the header-total semantics (see verify_D7_5_40_parity.py).
"""
import sys
from pathlib import Path
from collections import Counter
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
R = R.filter(pl.col("W_bomb").is_null() | (pl.col("W_bomb") > 0))

print("== per (period, round): bomb-anchored agreement (all player counts) ==")
B = R.filter(pl.col("W_bomb").is_not_null())
g = B.group_by(["period", "round"]).agg(
    pl.len().alias("n_bomb"),
    (pl.col("W_code") == pl.col("W_bomb")).sum().alias("code_eq"),
    (pl.col("W_within") == pl.col("W_bomb")).sum().alias("within_eq"),
    pl.col("W_within").is_not_null().sum().alias("n_within"),
    (pl.col("W_dp") == pl.col("W_bomb")).sum().alias("dp_eq"),
    pl.col("W_dp").is_not_null().sum().alias("n_dp"),
).sort(["period", "round"])
rows = g.rows()
for r in rows:
    p, rnd, n, ce, we, nw, de, nd = r
    if rnd >= 25 or rnd in (9, 13, 8, 12) or (rnd <= 2) or (ce != n):
        print(f"{p:4s} r{rnd:<3d} bomb_rows {n:5d} | code==bomb {ce:5d} | within==bomb {we:5d} of {nw:5d} | dp==bomb {de:5d} of {nd:5d}")
tot = B.group_by("period").agg(pl.len().alias("n"), (pl.col("W_code") == pl.col("W_bomb")).sum().alias("code_eq"),
                               (pl.col("W_within") == pl.col("W_bomb")).sum().alias("within_eq"),
                               pl.col("W_within").is_not_null().sum().alias("n_within"))
print("totals per period:", tot.rows())

print("\n== where code != within (both present): which label the bomb supports ==")
D = R.filter(pl.col("W_code").is_not_null() & pl.col("W_within").is_not_null() & (pl.col("W_code") != pl.col("W_within")))
print("rows:", D.height, "by period/round:", D.group_by(["period", "round"]).len().sort(["period", "round"]).rows())
Db = D.filter(pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0))
print("with bomb outcome:", Db.height)
print("bomb == code:", Db.filter(pl.col("W_bomb") == pl.col("W_code")).height, "| bomb == within:", Db.filter(pl.col("W_bomb") == pl.col("W_within")).height)
print("by period/round (bomb==code, bomb==within):",
      Db.group_by(["period", "round"]).agg((pl.col("W_bomb") == pl.col("W_code")).sum().alias("bomb_eq_code"),
                                          (pl.col("W_bomb") == pl.col("W_within")).sum().alias("bomb_eq_within"),
                                          pl.len().alias("n")).sort(["period", "round"]).rows())

print("\n== code != within, no bomb outcome: how many per period/round ==")
print(D.filter(pl.col("W_bomb").is_null()).group_by(["period", "round"]).len().sort(["period", "round"]).rows())

print("\n== reason_msg of code!=bomb rows (post) ==")
M = R.filter(pl.col("W_bomb").is_not_null() & (pl.col("W_bomb") > 0) & (pl.col("W_code") != pl.col("W_bomb")))
print(M.group_by(["period", "round", "W_bomb", "W_code", "reason_msg"]).len().sort(["period", "round"]).rows())

print("\n== header fields sanity (player_count==20 and all) ==")
print("rows with tF+cF == N (rounds):", R.with_columns(
    (pl.col("tF") + pl.col("cF") == pl.col("maxr")).alias("eq")).group_by(["period", "eq"]).len().sort(["period", "eq"]).rows())
