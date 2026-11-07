"""A4 probe: schemas and value distributions for the channels used by the bake-off round filters.

Read-only. Reads the 2026 tome basic channels with column projection.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A4_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"


def scan(channel):
    folder = TOME / f"basic_{channel}{SUFFIX}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.scan_parquet(files, extra_columns="ignore", missing_columns="insert")


for ch in ["round_end", "round_start", "round_state", "player_info", "player_disconnect",
           "bot_takeover", "player_death", "bomb_state", "team_change"]:
    lf = scan(ch)
    schema = lf.collect_schema()
    print(f"== {ch}: {dict(schema)}")
    df = lf.head(3).collect()
    print(df)

# value distributions that matter for the filters
re_ = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
print("round_end player_count distribution:")
print(re_.group_by("player_count").agg(pl.len().alias("rounds")).sort("player_count"))
print("round_end winner_team_code distribution:")
print(re_.group_by("winner_team_code").agg(pl.len().alias("rounds")).sort("winner_team_code"))
print("round_end null counts:", re_.null_count().row(0))

pi = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
print("player_info team_code distribution:")
print(pi.group_by("team_code").agg(pl.len().alias("rows")).sort("team_code"))
print("player_info null counts:", pi.null_count().row(0))

rs = scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
print("round_state event_type distribution:")
print(rs.group_by("event_type").agg(pl.len().alias("rows")).sort("rows", descending=True))
