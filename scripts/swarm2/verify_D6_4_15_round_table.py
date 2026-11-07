"""D6-4 verifier, step 15: which rounds carry round_end vs score disagreement (2024, player_count 20),
split by day window. Also the same count for bomb-known rows. Prints a compact table. Reads the step-2 join only.
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
j = j.with_columns(pl.when(pl.col("day").is_in([f"{x:02d}" for x in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("win"))
s = j.filter(pl.col("score_side").is_not_null())
t = (s.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
       .group_by(["win", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"))
       .sort(["win", "round"]))
print(t.filter(pl.col("disagree") > 0))
print("\nall rounds with score: rounds with any disagreement:", sorted(t.filter(pl.col("disagree") > 0)["round"].unique().to_list()))
bomb = s.filter(pl.col("bomb_side").is_not_null()).with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("dis"))
print("bomb-known rows, round_end != bomb, by round (d12-25 and d01-07 combined):")
print(bomb.filter(pl.col("dis")).group_by("round").len().sort("round").rows())
