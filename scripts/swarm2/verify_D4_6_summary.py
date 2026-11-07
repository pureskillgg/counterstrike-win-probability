"""D4-6 verifier summary. Re-computes the key numbers in one pass from the verifier table
(scratchpad parquet written by verify_D4_6_build.py from the 2024 tome).

ASSUMPTIONS (all marked):
  A1  64 ticks per second. 192 ticks = 3 s (A6 window in D4_elim_check), 429 ticks = 6.7 s.
  A2  Score side S from round_freeze_end(r) -> round_freeze_end(r+1) (same rule as D4).
  A3  Last death = max tick of deaths with team code 2/3 in [round_freeze_end, round_end]; single victim side required.
  A4  Full round = player_count 20 in the 2024 tome (10 players, 5 per side).
  A5  Days 12-25 = match_key date folder 12..25 (2024-02-12..25).
  A12 bomb_exploded => T (2) wins; bomb_defused => CT (3) wins.
  A16 Valid single-point score delta: exactly one side +1, other side +0.
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
               & (pl.col("n_victim_sides") == 1) & pl.col("victim_side").is_not_null()).with_columns(
    (pl.col("S") != pl.col("W")).alias("flag"),
    (pl.col("victim_side") == pl.col("W")).alias("victim_eq_W"),
    (5 - pl.col("victim_side") == pl.col("S")).alias("elim_eq_S"),
    pl.when(pl.col("has_bomb_out")).then(pl.lit("bomb")).otherwise(pl.lit("nobomb")).alias("kind"),
    pl.when(pl.col("gap") == 0).then(pl.lit("gap0")).when(pl.col("gap") == 429).then(pl.lit("gap429")).otherwise(pl.lit("gapOther")).alias("regime"))

print("== S1 reproduction of D4 claim (round, flag) with predicted-winner agreement (D4 'victim_eq_score')")
print(POP.filter(pl.col("round").is_in([9, 13, 25, 28])).group_by(["round", "flag"]).agg(
    pl.len().alias("n"), pl.col("elim_eq_S").mean().alias("P_elimwin_eq_S"), pl.col("victim_eq_W").mean().alias("P_victim_eq_W"),
    pl.col("gap").median().alias("median_gap")).sort(["round", "flag"]))

print("\n== S2 flagged set by round (POP): all rounds 1..30")
print(POP.group_by("round").agg(pl.len().alias("n"), pl.col("flag").sum().alias("n_flag")).filter(pl.col("n_flag") > 0).sort("round"))

print("\n== S3 bomb witness (no death rule, no score): bomb side vs S vs W, days 12-25, full, S!=0")
Bw = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0) & pl.col("has_bomb_out")).with_columns(
    (pl.col("S") != pl.col("W")).alias("flag"))
print(Bw.group_by(["round", "flag"]).agg(pl.len().alias("n"), (pl.col("bomb_side") == pl.col("S")).sum().alias("bomb_eq_S"),
                                         (pl.col("bomb_side") == pl.col("W")).sum().alias("bomb_eq_W")).filter(
    pl.col("round").is_in([2, 9, 13, 14, 25, 27, 28])).sort(["round", "flag"]))

print("\n== S4 round_end inversion check: flagged bomb rows, round_end winner vs bomb side")
print(Bw.filter(pl.col("flag")).group_by(["round", "bomb_side", "W"]).len().sort(["round", "bomb_side", "W"]))

print("\n== S5 gap regime (round 13, POP, non-bomb and bomb)")
print(POP.filter(pl.col("round") == 13).group_by("regime").agg(pl.len().alias("n"), pl.col("flag").mean().alias("P_flag")).sort("regime"))

print("\n== S6 3-second window (A6, gap <= 192): kept share and flag share, round 13 POP non-bomb")
r13 = POP.filter((pl.col("round") == 13) & (pl.col("kind") == "nobomb"))
print("rows", r13.height, " kept by 3s window", r13.filter(pl.col("gap") <= 192).height,
      " flagged rows kept", r13.filter((pl.col("gap") <= 192) & pl.col("flag")).height,
      " flagged rows total", r13.filter(pl.col("flag")).height)
print("P(flag | gap <= 192) =", r13.filter(pl.col("gap") <= 192).select(pl.col("flag").mean()).item())
print("P(elim == S | gap <= 192) =", r13.filter(pl.col("gap") <= 192).select(pl.col("elim_eq_S").mean()).item())

print("\n== S7 round 27 (score side artifact): tie status and agreement with round_end", )
F27 = R.filter((pl.col("round") == 27) & (pl.col("dd") >= 12) & (pl.col("player_count") == 20)).with_columns(
    (pl.col("t_score") == pl.col("ct_score")).alias("tied"))
print(F27.filter(pl.col("S") != 0).group_by("tied").agg(pl.len().alias("n_valid_S"), (pl.col("S") == pl.col("W")).sum().alias("S_eq_W")))
print("untied rows with S == 0:", F27.filter(~pl.col("tied") & (pl.col("S") == 0)).height, " of", F27.filter(~pl.col("tied")).height)
