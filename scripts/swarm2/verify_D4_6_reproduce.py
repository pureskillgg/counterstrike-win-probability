"""Reproduce the D4-6 claim numbers from the verifier table and check what each column actually measures.

Claim population (same as D4_flagged_rounds_death.py): 2024 days 12-25, full rounds (player_count 20),
score side S != 0, last death inside the live window [round_freeze_end, round_end] with a single victim side.
Flag = S != W (round_end winner_team_code).
Note: D4 computes pred_victim = 5 - victim_side. That is the ELIMINATION-IMPLIED WINNER, not the victim side.
This script reports both the literal victim-side agreement and the predicted-winner agreement.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
POP = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0)
               & (pl.col("n_victim_sides") == 1) & pl.col("victim_side").is_not_null())
POP = POP.with_columns(
    (pl.col("S") != pl.col("W")).alias("flag"),
    (pl.col("victim_side") == pl.col("S")).alias("victim_eq_S_literal"),
    (pl.col("victim_side") == pl.col("W")).alias("victim_eq_W"),
    (pl.col("victim_side") == (5 - pl.col("S"))).alias("victim_eq_opp_S"),
    (5 - pl.col("victim_side") == pl.col("S")).alias("elim_pred_winner_eq_S"),
    (5 - pl.col("victim_side") == pl.col("W")).alias("elim_pred_winner_eq_W"),
)
CLAIM_ROUNDS = [1, 2, 9, 10, 13, 14, 19, 25, 28]
print("POP rounds with a last death in window (2024 days 12-25, full rounds, S!=0):", POP.height)
print("\n== Claim rounds: flag split. Columns: P(victim==S) literal | P(victim==W) | P(elim-implied winner==S) (= D4 'victim_eq_score')")
t = (POP.filter(pl.col("round").is_in(CLAIM_ROUNDS))
     .group_by(["round", "flag"]).agg(
         pl.len().alias("n"),
         pl.col("victim_eq_S_literal").mean().alias("P_victim_eq_S"),
         pl.col("victim_eq_W").mean().alias("P_victim_eq_W"),
         pl.col("elim_pred_winner_eq_S").mean().alias("P_elimwin_eq_S"),
         pl.col("gap").median().alias("median_gap"),
         (pl.col("gap") == 429).mean().alias("share_gap_eq_429"),
         (pl.col("gap") == 0).mean().alias("share_gap_eq_0"),
     ).sort(["round", "flag"]))
print(t)

print("\n== Flag rate among POP rounds, by round number (all rounds 1..30, days 12-25)")
FL = POP.group_by("round").agg(pl.len().alias("n"), pl.col("flag").sum().alias("n_flag"),
                               pl.col("flag").mean().alias("flag_rate")).sort("round")
print(FL)
