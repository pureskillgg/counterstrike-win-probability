"""Copy-of-previous-round test for the round_end winner. Hypothesis (from two timelines): round_end(r).winner_team_code
equals round_end(r-1).winner_team_code on flagged rows, i.e. the flagged round_end is a stale copy of the previous round.

  K1. Flagged rows (S != W, days 12-25, full): share with W(r) == W(r-1), and W(r) == S(r-1), and S(r) == W(r-1).
  K2. Same shares for unflagged rows (control).
  K3. Round-by-round for the claim rounds and round 27.
  K4. Bomb witness on round 13 flagged rows vs W(12): does the bomb side equal the stale previous winner?
ASSUMPTIONS: A14. Previous round = round r-1 in the same match (round numbering from the tome; checked for gaps).
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
P = R.select(["match_key", "round", pl.col("W").alias("W_prev"), pl.col("S").alias("S_prev"), pl.col("bomb_side").alias("bomb_prev")]).with_columns(
    (pl.col("round") + 1).alias("round"))
X = R.join(P, on=["match_key", "round"], how="left").filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & (pl.col("S") != 0))
X = X.with_columns((pl.col("S") != pl.col("W")).alias("flag"))
print("K1/K2: share of rows with W(r)==W(r-1), S(r)==W(r-1), W(r)==S(r-1), by round, flagged vs unflagged")
K = X.filter(pl.col("round").is_in([9, 13, 25, 27, 28, 2, 10, 14, 19, 24, 26, 29]) & pl.col("W_prev").is_not_null()).group_by(["round", "flag"]).agg(
    pl.len().alias("n"),
    (pl.col("W") == pl.col("W_prev")).mean().alias("P_W_eq_Wprev"),
    (pl.col("S") == pl.col("W_prev")).mean().alias("P_S_eq_Wprev"),
    (pl.col("S_prev") == pl.col("W")).mean().alias("P_Wnow_eq_Sprev"),
).sort(["round", "flag"])
print(K)
print("\nK4: round 13 flagged rows with bomb outcome: bomb side vs W(12) (stale copy?) vs S(13)")
B13 = X.filter((pl.col("round") == 13) & pl.col("flag") & pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != 0))
print("n", B13.height, " bomb==W_prev:", B13.select((pl.col("bomb_side") == pl.col("W_prev")).mean()).item(),
      " bomb==S:", B13.select((pl.col("bomb_side") == pl.col("S")).mean()).item())
print("day split, flagged round 13: P(W==W_prev), n:")
print(X.filter((pl.col("round") == 13) & pl.col("flag") & pl.col("W_prev").is_not_null()).group_by("dd").agg(
    pl.len().alias("n"), (pl.col("W") == pl.col("W_prev")).mean().alias("P_W_eq_Wprev")).sort("dd"))
print("\nK5: all 2024 round 13 rows (days 1-25, full, incl S=0), P(W==W_prev) by day:")
A13 = R.join(P, on=["match_key", "round"], how="left").filter((pl.col("round") == 13) & (pl.col("player_count") == 20) & pl.col("W_prev").is_not_null())
print(A13.group_by("dd").agg(pl.len().alias("n"), (pl.col("W") == pl.col("W_prev")).mean().alias("P_W_eq_Wprev"),
                             (pl.col("S") != pl.col("W")).mean().alias("P_flag")).sort("dd"))
