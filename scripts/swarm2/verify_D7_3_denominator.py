"""verify D7-3: reproduce the claimed 605 / 434 round-13 bomb denominators by folder day, and the build split. Read-only."""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
S = pl.read_parquet(str(SCR / "verify_D7_3_score_rounds.parquet"))
r13b = S.filter((pl.col("round") == 13) & pl.col("has_bomb")).with_columns(
    (pl.col("day").cast(pl.Int64) >= 12).alias("folder_12_25"))
print("round-13 rows with a bomb outcome, all builds:", r13b.height)
print("by build x folder_12_25 (rows, round_end != bomb):")
print(r13b.group_by(["build_num", "folder_12_25"]).agg(pl.len().alias("n"),
                                                        (pl.col("winner_team_code") != pl.col("bomb_side")).sum().alias("mm"))
      .sort(["build_num", "folder_12_25"]))
sub = r13b.filter(pl.col("folder_12_25"))
print("folder 12-25 subset, all builds: n =", sub.height, "| mismatches =", sub.filter(pl.col("winner_team_code") != pl.col("bomb_side")).height)
print("folder 12-25 subset with player_count == 20: n =", sub.filter(pl.col("player_count") == 20).height)
print("build 9961 only, all folders: n =", r13b.filter(pl.col("build_num") == 9961).height,
      "| mismatches =", r13b.filter((pl.col("build_num") == 9961) & (pl.col("winner_team_code") != pl.col("bomb_side"))).height)
