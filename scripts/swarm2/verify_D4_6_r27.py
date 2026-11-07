"""Round 27 and the side-swap boundary question.

  R1. Planter side by round (days 12-25, all matches): a planted bomb should have planter side 2 (T). Count exceptions.
  R2. Scores around the 27->28 boundary (round_freeze_end 26, 27, 28 and round_officially_ended 27) for a few
      flagged round-27 matches: does the score pair swap sides between freeze 27 and freeze 28?
  R3. Swap-boundary test over all matches: P(t_score(r+1) == ct_score(r) and ct_score(r+1) == t_score(r)) at each round
      boundary, to locate where side swaps occur (halftime at 12/13, OT at 24/25, 27/28).
  R4. Death-implied winner (5 - victim side) for round 27 and 28 flagged rows: which side does it match?
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
import glob as _g


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(_g.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


bs = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
pl_ = bs.filter(pl.col("event_type") == "bomb_planted").with_columns(
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"))
pl_ = pl_.filter(pl.col("day").cast(pl.Int32) >= 12)
print("R1 planter side by round (days 12-25, bomb_planted rows):")
print(pl_.group_by(["round", "player_team_code"]).len().sort(["round", "player_team_code"]).filter(
    pl.col("round").is_in([9, 12, 13, 24, 25, 26, 27, 28, 29, 30])))

rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fz = rs.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"]).unique(
    ["match_key", "round"], keep="first").select(["match_key", "round", "t_score", "ct_score"])
ro = rs.filter(pl.col("event_type") == "round_officially_ended").sort(["match_key", "round", "tick"]).unique(
    ["match_key", "round"], keep="first").select(["match_key", "round", pl.col("t_score").alias("t_end"),
                                                 pl.col("ct_score").alias("ct_end")])
Z = fz.join(ro, on=["match_key", "round"], how="left")
nx = fz.with_columns((pl.col("round") - 1).alias("round")).select(["match_key", "round", pl.col("t_score").alias("t_n"),
                                                                    pl.col("ct_score").alias("ct_n")])
Bd = fz.join(nx, on=["match_key", "round"], how="inner")
Bd = Bd.with_columns(
    ((pl.col("t_n") == pl.col("ct_score")) & (pl.col("ct_n") == pl.col("t_score"))).alias("swapped_pattern"),
    ((pl.col("t_n") == pl.col("t_score")) & (pl.col("ct_n") == pl.col("ct_score"))).alias("same_pattern"))
print("\nR3 boundary pattern (freeze r -> freeze r+1), day 12-25, by round r:")
Bd = Bd.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("dd")).filter(
    pl.col("dd") >= 12)
print(Bd.group_by("round").agg(pl.len().alias("n"), pl.col("swapped_pattern").mean().alias("P_swapped"),
                              pl.col("same_pattern").mean().alias("P_same")).filter(
    pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29])).sort("round"))

R = pl.read_parquet(SCRATCH / "verify_D4_6_rounds_2024.parquet")
F27 = R.filter((pl.col("round") == 27) & (pl.col("dd") >= 12) & (pl.col("S") != 0) & (pl.col("player_count") == 20))
print("\nR4 round 27 rows (S!=0, full, days 12-25):", F27.height, " flagged:", F27.filter(pl.col("S") != pl.col("W")).height)
print(F27.select([pl.col("S").alias("S"), "W", "win_reason_code", "victim_side", "n_victim_sides", "gap"]).group_by(
    ["S", "W", "victim_side", "win_reason_code"]).len().sort("len", descending=True).head(12))
print("R4 round 27: death-implied winner (5-victim) == S share:",
      F27.filter(pl.col("victim_side").is_not_null()).select((5 - pl.col("victim_side") == pl.col("S")).mean()).item())
print("R4 round 27: death-implied winner (5-victim) == W share:",
      F27.filter(pl.col("victim_side").is_not_null()).select((5 - pl.col("victim_side") == pl.col("W")).mean()).item())
