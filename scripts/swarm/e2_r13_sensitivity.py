"""Sensitivity for the 2024 round-13 side-code defect found by e2_side_check_r13.py.

Imports the E2 loader (scripts\\swarm\\e2_map_profile.py, unchanged) and recomputes, per map for 2024:
  - CT win rate on all clean rounds (should match the E2 table)
  - CT win rate excluding round 13
  - round-13 CT win rate and n
Also: round-13 bomb-outcome mismatch by day (from the tome), and round_end win_reason_message vs winner code at
round 13 (internal consistency check inside round_end itself).
Run: pixi run python -I -X utf8 scripts\\swarm\\e2_r13_sensitivity.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

HERE = Path(r"G:\pureskill\csds-cs2\scripts\swarm")
sys.path.insert(0, str(HERE))
import e2_map_profile as E2  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
cfg = E2.ERAS["2024"]
hdr, rounds, d, groups = E2.load_era("2024", cfg)
rounds = rounds.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

print("2024 clean rounds:", rounds.height, "matches:", rounds["match_key"].n_unique())
per_map = rounds.group_by("map_name").agg(
    pl.len().alias("rounds_all"),
    pl.col("ct_win").mean().alias("ct_all"),
    (pl.col("round") != 13).sum().alias("rounds_ex13"),
    pl.col("ct_win").filter(pl.col("round") != 13).mean().alias("ct_ex13"),
    (pl.col("round") == 13).sum().alias("r13_n"),
    pl.col("ct_win").filter(pl.col("round") == 13).mean().alias("ct_r13"),
).sort("rounds_all", descending=True)
with pl.Config(tbl_rows=-1, tbl_cols=-1):
    print(per_map)

all_ex13 = rounds.filter(pl.col("round") != 13)
print(f"\nall maps: CT all {rounds['ct_win'].mean():.4f}, CT excl r13 {all_ex13['ct_win'].mean():.4f}, "
      f"r13 n {int((rounds['round'] == 13).sum())}, r13 CT {rounds.filter(pl.col('round') == 13)['ct_win'].mean():.4f}")

r13 = rounds.filter(pl.col("round") == 13)
print("\nround-13 CT win rate by day (2024, clean rounds):")
with pl.Config(tbl_rows=-1):
    print(r13.group_by("day").agg(pl.len().alias("n"), pl.col("ct_win").mean().alias("ct_rate")).sort("day"))

# round_end internal check: win_reason_message vs winner code, round 13 vs other rounds
files = sorted(glob.glob(str(cfg["base"] / f"basic_round_end{cfg['suffix']}" / "dataframe_*")))
RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "win_reason_message",
                                             "player_count"]) for f in files], how="vertical_relaxed")
RE = RE.filter(pl.col("player_count") == 20).with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("bin"))
print("\nround_end win_reason_message x winner_team_code (2024, clean 20-player rounds):")
with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=40):
    print(RE.group_by(["bin", "win_reason_message", "winner_team_code"]).len().sort(["bin", "len"], descending=[False, True]))
