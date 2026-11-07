"""Final checks for D4-6.
  F1. Claim's 'round-13 no-bomb flagged matches (825), last death = score side 96.8%': reproduce the population.
      Our population: flagged round-13 rows with a last death in window, split by bomb outcome.
  F2. Bomb-outcome rows for round 13: bomb tick vs round_end tick by gap regime (is the round_end lag also present on bomb rounds?).
  F3. Flag rate by gap regime for bomb rounds in 9, 13, 25, 28.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
POP = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0)
               & (pl.col("n_victim_sides") == 1) & pl.col("victim_side").is_not_null()).with_columns(
    (pl.col("S") != pl.col("W")).alias("flag"), (5 - pl.col("victim_side") == pl.col("S")).alias("elim_eq_S"),
    pl.when(pl.col("gap") == 0).then(pl.lit("gap0")).when(pl.col("gap") == 429).then(pl.lit("gap429")).otherwise(pl.lit("gapOther")).alias("regime"))
F = POP.filter((pl.col("round") == 13) & pl.col("flag"))
print("F1 flagged round-13 rows (POP):", F.height)
print(F.group_by(pl.col("has_bomb_out")).agg(pl.len().alias("n"), pl.col("elim_eq_S").mean().alias("P_elim_eq_S")))
print("F1 no-bomb flagged round-13 rows: P(elim==S) by win reason:")
print(F.filter(~pl.col("has_bomb_out")).group_by("win_reason_code").agg(pl.len().alias("n"),
                                                                       pl.col("elim_eq_S").mean().alias("P_elim_eq_S")).sort("win_reason_code"))
print("\nF2 round-13 bomb rows (POP): bomb tick vs round_end tick by gap regime")
Bq = POP.filter((pl.col("round") == 13) & pl.col("has_bomb_out")).with_columns(
    (pl.col("end") - pl.col("bomb_out_tick")).alias("end_minus_bomb"))
print(Bq.group_by("regime").agg(pl.len().alias("n"), pl.col("flag").mean().alias("P_flag"),
                                pl.col("end_minus_bomb").median().alias("median_end_minus_bomb"),
                                (pl.col("end_minus_bomb") == 0).mean().alias("P_end_eq_bomb")))
print("\nF3 bomb rounds by round (POP): flag rate by gap regime")
print(POP.filter(pl.col("has_bomb_out") & pl.col("round").is_in([9, 13, 25, 27, 28])).group_by(["round", "regime"]).agg(
    pl.len().alias("n"), pl.col("flag").mean().alias("P_flag")).sort(["round", "regime"]))
