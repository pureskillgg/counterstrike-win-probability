"""verify_D9_9 step 3: score-stream winner oracle (independent of round_end winner, bomb events and death labels).

Score-stream winner per round:
  - straight delta (t_after - t_prev, ct_after - ct_prev) is a unit vector -> T (2) if dT == 1 else CT (3)
  - swap-signature round (straight not unit, swapped pairing unit): event labels are post-swap, so the side
    that was T during the round is the CT-labelled score; winner = T (2) if (ct_after - t_prev) == 1 else CT (3)
  - otherwise null.
ASSUMPTION: round_state t_score / ct_score are side-keyed scores that swap at the swap boundary.
Compares: score winner vs recorded round_end winner vs bomb truth, by build x round class.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round.parquet")


def unit(a, b):
    return ((a == 1) & (b == 0)) | ((a == 0) & (b == 1))


R = R.with_columns(
    unit(pl.col("dT_straight"), pl.col("dCT_straight")).alias("su"),
    unit(pl.col("dT_swapped"), pl.col("dCT_swapped")).alias("wu"),
)
# swap-signature per match-round, using the same definition as step 2c
R = R.with_columns(((~pl.col("su")) & pl.col("wu") & pl.col("t_after").is_not_null()).alias("swap_sig"))
R = R.with_columns(
    pl.when(pl.col("t_after").is_null()).then(None)
      .when(pl.col("swap_sig")).then(pl.when(pl.col("dCT_swapped") == 1).then(2).otherwise(3))
      .when(pl.col("su")).then(pl.when(pl.col("dT_straight") == 1).then(2).otherwise(3))
      .otherwise(None).alias("w_score"))
R = R.with_columns(
    (pl.col("w_score") == pl.col("winner_team_code")).alias("score_eq_recorded"),
    pl.when(pl.col("w_score").is_null()).then(None).otherwise(pl.col("w_score") == pl.col("bomb_side")).alias("score_eq_bomb"),
)
R.write_parquet(SCR / "verify_D9_9_round_oracle.parquet")

print("== coverage of score winner: rounds, null score winner")
print(R.group_by("build_num").agg(pl.len().alias("rounds"), pl.col("w_score").is_null().sum().alias("null_score"),
                                 pl.col("swap_sig").sum().alias("swap_sig_rounds")))

print("\n== score winner vs recorded round_end winner, by build x round (rounds 8-14, 24-29)")
g = R.filter(pl.col("w_score").is_not_null() & pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 24, 25, 26, 27, 28, 29]))
print(g.group_by(["build_num", "round"]).agg(pl.len().alias("n"),
                                             pl.col("score_eq_recorded").mean().round(3).alias("frac_score_eq_recorded")
                                             ).sort(["build_num", "round"]))

print("\n== bomb-outcome rounds: score winner vs bomb truth and recorded vs bomb truth, by build x round class")
bo = R.filter(pl.col("has_outcome") & pl.col("w_score").is_not_null())
print(bo.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("score_eq_bomb").mean().round(4).alias("score_eq_bomb"),
    (pl.col("winner_team_code") == pl.col("bomb_side")).mean().round(4).alias("recorded_eq_bomb"),
    pl.col("score_eq_recorded").mean().round(4).alias("score_eq_recorded")).sort(["build_num", "rclass"]))
print("\n== bomb-outcome rounds build 9961 round 13 (pc20 or all): score vs bomb vs recorded counts")
b13 = bo.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13))
print(b13.group_by(["score_eq_bomb", "bomb_mismatch"]).len().sort(["score_eq_bomb", "bomb_mismatch"]))

print("\n== non-bomb rounds: score winner vs recorded, by build x round class")
nb = R.filter(~pl.col("has_outcome") & pl.col("w_score").is_not_null() & pl.col("n_CT").is_not_null())
print(nb.group_by(["build_num", "rclass"]).agg(pl.len().alias("n"),
                                                pl.col("score_eq_recorded").mean().round(4).alias("score_eq_recorded")
                                                ).sort(["build_num", "rclass"]))
