"""verify D1-7 (swarm2 verifier): does the winner_team_code bomb mismatch at round r follow an in-window exchange at round r-1 in the same match?

ASSUMPTIONS: as in verify_D1_7_main.py (A1-A7). Exchange = n_swap > 0 in the saved per-round table.
Rounds tested: 9, 13, 25, 28 (2024 d12-25 winner mismatches) and 13, 28 in 2026 as a control.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")

J24 = pl.read_parquet(OUT / "verify_D1_7_rounds2024.parquet")
J26 = pl.read_parquet(OUT / "verify_D1_7_rounds2026.parquet")


def add_prev(J):
    prev = J.select(["match_key", "round", "n_swap"]).with_columns((pl.col("round") + 1).alias("round"),
                                                                    (pl.col("n_swap") > 0).alias("prev_exch"))
    prev = prev.select(["match_key", "round", "prev_exch"])
    return J.join(prev, on=["match_key", "round"], how="left").with_columns(pl.col("prev_exch").fill_null(False))


J24 = add_prev(J24)
J26 = add_prev(J26)
J24 = J24.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
J24 = J24.with_columns(pl.when(pl.col("day") >= 12).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

print("== 2024 d12-25: winner != bomb, by round and previous-round in-window exchange")
x = J24.filter((pl.col("dg") == "d12-25") & pl.col("bomb_side").is_not_null() & pl.col("round").is_in([9, 13, 25, 28]))
print(x.group_by(["round", "prev_exch"]).agg(
    pl.len().alias("bomb_rows"),
    pl.col("bomb_mm_winner").sum().alias("winner_ne_bomb"),
    (pl.col("bomb_side") == pl.col("winner_team_code")).sum().alias("winner_eq_bomb")).sort(["round", "prev_exch"]))

print("\n== 2024 d12-25 winner != bomb at ANY round, by prev_exch (rounds 2..30)")
y = J24.filter((pl.col("dg") == "d12-25") & pl.col("bomb_side").is_not_null() & pl.col("round").is_between(2, 30))
print(y.group_by("prev_exch").agg(pl.len().alias("bomb_rows"), pl.col("bomb_mm_winner").sum().alias("winner_ne_bomb")).sort("prev_exch").to_dicts())
print("winner != bomb rounds:", y.filter(pl.col("bomb_mm_winner")).group_by("round").len().sort("round").to_dicts())

print("\n== 2024 d01-07 winner != bomb (any round, rounds 2..30):",
      int(J24.filter((pl.col("dg") == "d01-07") & pl.col("bomb_side").is_not_null() & pl.col("round").is_between(2, 30))["bomb_mm_winner"].sum()))

print("\n== 2026 control: winner != bomb by round and prev_exch (rounds 9, 13, 25, 28, 34, 40)")
z = J26.filter(pl.col("bomb_side").is_not_null() & pl.col("round").is_in([9, 13, 25, 28, 34, 40]))
print(z.group_by(["round", "prev_exch"]).agg(pl.len().alias("bomb_rows"), pl.col("bomb_mm_winner").sum().alias("winner_ne_bomb")).sort(["round", "prev_exch"]))
print("2026 all bomb rows:", int(J26.filter(pl.col("bomb_side").is_not_null()).height),
      " winner != bomb:", int(J26.filter(pl.col("bomb_side").is_not_null())["bomb_mm_winner"].sum()))
