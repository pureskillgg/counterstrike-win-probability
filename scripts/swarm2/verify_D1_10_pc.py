"""D1-10 verifier: player_count populations. Is the 2026 zero comparing the same kind of rounds as 2024?

Tabulates round_end.player_count in the 2026 and 2024 tomes, then the 2024 claim-rule and bomb conflicts
by player_count (all rounds, and round 13 only). Read-only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def fl(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


re26 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]).collect() for f in fl(T26, "round_end")]).unique(["match_key", "round"])
re24 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]).collect() for f in fl(T24, "round_end")]).unique(["match_key", "round"])
print("2026 round_end by player_count (top 12):", sorted(re26.group_by("player_count").len().rows(), key=lambda x: -x[1])[:12])
print("2024 round_end by player_count (top 12):", sorted(re24.group_by("player_count").len().rows(), key=lambda x: -x[1])[:12])
print("2026 pc==20 rounds:", re26.filter(pl.col("player_count") == 20).height, "  2024 pc==20 rounds:", re24.filter(pl.col("player_count") == 20).height)

RS24 = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect() for f in fl(T24, "round_state")])
s = RS24.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t_b"), pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
    pl.col("t_score").sort_by("tick").last().alias("t_a"), pl.col("ct_score").sort_by("tick").last().alias("ct_a"))
s = s.with_columns(
    pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
      .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))
bs = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]).collect() for f in fl(T24, "bomb_state")])
bo = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).group_by(["match_key", "round"])
      .agg((pl.col("event_type") == "bomb_exploded").any().alias("e"), (pl.col("event_type") == "bomb_defused").any().alias("f")))
bo = bo.with_columns(pl.when(pl.col("e") & ~pl.col("f")).then(pl.lit(2)).when(pl.col("f") & ~pl.col("e")).then(pl.lit(3))
                     .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
J = re24.join(s.select(["match_key", "round", "claim_side"]), on=["match_key", "round"], how="left").join(
    bo.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
J = J.with_columns((pl.col("claim_side") != pl.col("winner_team_code")).alias("dis_claim"),
                   (pl.col("bomb_side") != pl.col("winner_team_code")).alias("dis_bomb"))
print("\n2024 claim-rule disagreements by player_count (top 10):",
      sorted(J.filter(pl.col("dis_claim")).group_by("player_count").len().rows(), key=lambda x: -x[1])[:10])
print("2024 bomb-vs-winner conflicts by player_count (top 10):",
      sorted(J.filter(pl.col("bomb_side").is_not_null() & pl.col("dis_bomb")).group_by("player_count").len().rows(), key=lambda x: -x[1])[:10])
r13 = J.filter(pl.col("round") == 13)
print("2024 round 13 bomb conflicts by player_count:",
      sorted(r13.filter(pl.col("bomb_side").is_not_null() & pl.col("dis_bomb")).group_by("player_count").len().rows(), key=lambda x: -x[1])[:10])
print("2024 round 13 bomb-labelled rows by player_count:",
      sorted(r13.filter(pl.col("bomb_side").is_not_null()).group_by("player_count").len().rows(), key=lambda x: -x[1])[:10])
print("DONE")
