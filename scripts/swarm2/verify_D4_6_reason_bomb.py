"""Win reason vs bomb outcome for round 13 and control rounds (days 12-25, full rounds, all rows incl. S=0).
Game rule: bomb_exploded -> reason 1 (Target_Bombed, T wins); bomb_defused -> reason 7 (Bomb_Defused, CT wins).
Crosstab bomb outcome x win reason, split by flag (S != W). If flagged bomb rows carry elimination reasons (8/9), the
round_end record itself is inconsistent with the bomb events in those rows.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
X = R.filter((pl.col("player_count") == 20) & (pl.col("dd") >= 12) & pl.col("has_bomb_out")).with_columns(
    pl.when(pl.col("bomb_side") == 2).then(pl.lit("exploded")).otherwise(pl.lit("defused")).alias("bomb_out"),
    pl.when((pl.col("S") != pl.col("W")) & (pl.col("S") != 0)).then(pl.lit("flag")).otherwise(pl.lit("agree_or_S0")).alias("status"))
print("bomb outcome x win reason, round 13 (days 12-25):")
print(X.filter(pl.col("round") == 13).group_by(["status", "bomb_out", "win_reason_code"]).len().sort(["status", "bomb_out", "win_reason_code"]))
print("\nbomb outcome x win reason, rounds 2, 10, 14, 19 (controls, days 12-25):")
print(X.filter(pl.col("round").is_in([2, 10, 14, 19])).group_by(["bomb_out", "win_reason_code"]).len().sort(["bomb_out", "win_reason_code"]))
print("\nbomb outcome x win reason, rounds 9, 25, 27, 28:")
print(X.filter(pl.col("round").is_in([9, 25, 27, 28])).group_by(["round", "status", "bomb_out", "win_reason_code"]).len().sort(["round", "status", "bomb_out", "win_reason_code"]))
