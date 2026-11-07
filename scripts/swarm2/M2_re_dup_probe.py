"""M2 probe: 2024 round_end duplicates. Per (match, round) row counts, player_count and legacy_code patterns."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
ROOT = Path("G:/pureskill/csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
fd = BASE / f"basic_round_end{SUF}"
re_ = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(fd / "dataframe_*")))], how="vertical_relaxed")
print("rows", re_.height)
cnt = re_.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("player_count").n_unique().alias("npc"),
                                                pl.col("winner_team_code").n_unique().alias("nwin"),
                                                pl.col("tick").n_unique().alias("ntick"))
print("rows per (match, round):", cnt.group_by("n").agg(pl.len()).sort("n").rows())
print("rounds with >1 distinct winner:", cnt.filter(pl.col("nwin") > 1).height)
print("rounds with >1 distinct tick:", cnt.filter(pl.col("ntick") > 1).height)
print("rounds with >1 distinct player_count:", cnt.filter(pl.col("npc") > 1).height)
print("player_count x rows-per-round:", cnt.group_by(["npc"]).agg(pl.len()).rows())
print("player_count value counts:", re_.group_by("player_count").agg(pl.len()).sort("player_count").rows())
print("legacy_code value counts:", re_.group_by("legacy_code").agg(pl.len()).sort("legacy_code").rows())
print("legacy_code x player_count:", re_.group_by(["legacy_code", "player_count"]).agg(pl.len()).sort(["legacy_code", "player_count"]).head(40).rows())
print("win_reason_message sample:", re_.group_by("win_reason_message").agg(pl.len()).sort("len", descending=True).head(15).rows())
one = re_.filter(pl.col("match_key") == re_["match_key"][0]).sort(["round", "tick"]).head(40)
print(one.select(["round", "tick", "winner_team_code", "win_reason_code", "legacy_code", "player_count"]).rows())
