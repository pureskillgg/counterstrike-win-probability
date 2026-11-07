"""verify D7-6 step c: coverage of player-level side codes per (match, round) in the 2024 tome.
Tables: player_info (round, player_id, team_code), player_spawn (round, player_id, team_code), player_death (victim team code).
Question: can per-round physical side (T=2 / CT=3) be read for all 20 players, independent of score fields?
Stdout only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def scan(name, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{name}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs).select(cols).collect()


RE = scan("round_end", ["match_key", "round", "player_count"]).unique(["match_key", "round"], keep="first")
PI = scan("player_info", ["match_key", "round", "player_id", "team_code"])
print("player_info rows:", PI.height, "| distinct rounds:", PI.select(["match_key", "round"]).unique().height)
print("player_info team_code dist:", PI.group_by("team_code").len().sort("team_code").rows())
print("player_info round values (top):", PI.group_by("round").len().sort("len", descending=True).head(8).rows())
pir = (PI.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("np"),
       (pl.col("team_code") == 2).sum().alias("nT"), (pl.col("team_code") == 3).sum().alias("nCT")))
print("player_info per (match,round): np dist:", pir.group_by("np").len().sort("np").head(12).rows())
print("player_info per (match,round): nT,nCT both 10:", pir.filter((pl.col("nT") == 10) & (pl.col("nCT") == 10)).height, "of", pir.height)

PS = scan("player_spawn", ["match_key", "round", "player_id", "player_team_code", "tick"])
psr = PS.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("np"))
print("player_spawn rounds:", psr.height, "| np dist (top):", psr.group_by("np").len().sort("np").head(12).rows())

PD = scan("player_death", ["match_key", "round", "player_id", "player_team_code", "tick"])
print("player_death rows:", PD.height, "| null player_team_code:", PD.filter(pl.col("player_team_code").is_null()).height)
print("player_death team codes:", PD.group_by("player_team_code").len().sort("player_team_code").rows())
pdr = PD.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("nd"))
rr = RE.join(pdr, on=["match_key", "round"], how="left").with_columns(pl.col("nd").fill_null(0))
print("rounds (round_end) with 20 players:", RE.filter(pl.col("player_count") == 20).height)
print("death count per round (20-player rounds) dist, top:",
      rr.filter(pl.col("player_count") == 20).group_by("nd").len().sort("nd").head(25).rows())
