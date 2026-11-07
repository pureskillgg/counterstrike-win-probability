"""D9 probe 7: build_num vs game_build crosstab (matches) and round-index coverage for build 9961 (pc20)."""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")

print("== build_num x game_build (matches)")
print(M.group_by(["build_num", "game_build"]).len().sort(["build_num", "game_build"]))

print("\n== build 9961, pc20, outcome rounds: round index coverage (rounds >= 26 and round totals)")
B = R.join(M.select(["match_key", "build_num"]), on="match_key", how="left").filter(
    pl.col("has_outcome") & (pl.col("player_count") == 20) & (pl.col("build_num") == 9961))
print("   max round index in build 9961 pc20 outcome rounds:", B["round"].max())
print("   outcome rounds with round >= 26:", B.filter(pl.col("round") >= 26).height,
      "mismatch:", int(B.filter(pl.col("round") >= 26)["mismatch"].sum()))
print("   outcome rounds by round index (9961, pc20), all:",
      B.group_by("round").agg(pl.len().alias("n")).sort("round").filter(pl.col("round").is_in([9, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30])).rows())
print("   build 9961 matches with any round >= 25:",
      R.join(M.select(["match_key", "build_num"]), on="match_key", how="left").filter(
          (pl.col("build_num") == 9961) & (pl.col("round") >= 25)).select("match_key").n_unique())
