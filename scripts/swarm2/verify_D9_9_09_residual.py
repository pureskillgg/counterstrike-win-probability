"""verify_D9_9 step 9: residual score-vs-recorded disagreements in build 9961 outside the half-start positions.

Counts rounds with a score winner where recorded != score, excluding the half-start positions
(5v5 13/25/28, 2v2 9), grouped by round index; and checks whether the count is concentrated at round 27
(the known score-rule issue at the OT switch) or elsewhere.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round_oracle.parquet")
fmt = (R.group_by("match_key").agg(pl.col("n_T").median().alias("nT_med"))
       .with_columns(pl.when(pl.col("nT_med") <= 2).then(pl.lit("2v2")).otherwise(pl.lit("5v5")).alias("fmt")))
R = R.join(fmt.select(["match_key", "fmt"]), on="match_key", how="left")
half = (((pl.col("fmt") == "5v5") & pl.col("round").is_in([13, 25, 28])) | ((pl.col("fmt") == "2v2") & (pl.col("round") == 9)))
S = R.filter(pl.col("w_score").is_not_null() & (pl.col("build_num") == 9961) & ~half)
dis = S.filter(pl.col("w_score") != pl.col("winner_team_code"))
print("9961 rounds with score winner outside half positions:", S.height, "| disagreements:", dis.height)
print("disagreements by round index (top):")
print(dis.group_by("round").len().sort("len", descending=True).head(15))
print("disagreements by format:", dis.group_by("fmt").len().rows())
S8 = R.filter(pl.col("w_score").is_not_null() & (pl.col("build_num") == 9842))
print("9842 rounds with score winner:", S8.height, "| disagreements:", S8.filter(pl.col("w_score") != pl.col("winner_team_code")).height)
print("9842 disagreements by round:", S8.filter(pl.col("w_score") != pl.col("winner_team_code")).group_by("round").len().sort("round").rows())
