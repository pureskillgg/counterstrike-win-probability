"""Independent checks on data\\state_sample outputs (reads only those two files).

Checks the snapshot grid and keys from the saved tables, not from the extractor's code path:
  - snapshot tick = live start + 64 * seconds_since_live, and tick < round_end tick
  - rounds: unique keys; every player row belongs to a round; per round the number of distinct
    snapshot ticks equals the grid count sum(ceil((end - live) / 64)) of the rounds table
  - per (match, round, tick, player_id) unique; team_code in {2, 3}; health and armor 0..100
  - each match has one split; plant tick null or inside the live window
  - ct_score + t_score == round - 1 at live start (bake-off score check)
  - players per side per snapshot at most 5

Run from the project root:
  pixi run python -I -X utf8 scripts\\eda\\verify_state_1hz.py
"""

import math
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
P = ROOT / "data" / "state_sample" / "players_1hz.parquet"
R = ROOT / "data" / "state_sample" / "rounds.parquet"


def main():
    pl_ = pl.read_parquet(P)
    rd = pl.read_parquet(R)
    print(f"players_1hz: {pl_.height} rows, {pl_.width} columns; rounds: {rd.height} rows, {rd.width} columns")
    print("players columns:", pl_.columns)
    print("rounds columns:", rd.columns)

    # keys and rounds
    dup_rounds = rd.height - rd.unique(["match_key", "round"]).height
    print(f"rounds unique by (match_key, round): duplicates {dup_rounds}")
    j = pl_.join(rd.select(["match_key", "round", "live_start_tick", "round_end_tick", "split",
                            "ct_score", "t_score", "plant_tick", "ct_win"]),
                 on=["match_key", "round"], how="left")
    print(f"player rows without a rounds row: {j.filter(pl.col('split').is_null()).height}")
    off_grid = j.filter((pl.col("tick") - pl.col("live_start_tick")) != 64 * pl.col("seconds_since_live"))
    past_end = j.filter(pl.col("tick") >= pl.col("round_end_tick"))
    print(f"rows off the 64-tick grid: {off_grid.height}; rows at or after round_end: {past_end.height}")
    dup_players = pl_.height - pl_.unique(["match_key", "round", "tick", "player_id"]).height
    print(f"duplicate (match, round, tick, player_id) rows: {dup_players}")
    print(f"team_code values: {sorted(pl_['team_code'].unique().to_list())}")
    print(f"health range {pl_['health'].min()}..{pl_['health'].max()}; armor range "
          f"{pl_['armor'].min()}..{pl_['armor'].max()}")
    print(f"nulls in player columns: " + str({c: pl_[c].null_count() for c in pl_.columns if pl_[c].null_count()}))

    # grid count: every snapshot in the rounds table should appear in the players table
    grid = sum(math.ceil((e - s) / 64) for s, e in rd.select(["live_start_tick", "round_end_tick"]).rows())
    snaps_present = pl_.select(["match_key", "round", "tick"]).unique().height
    print(f"snapshot grid count from rounds table: {grid}; distinct snapshots in players table: {snaps_present}")

    # per-snapshot side counts
    side = pl_.group_by(["match_key", "round", "tick"]).agg(
        (pl.col("team_code") == 2).sum().alias("nT"), (pl.col("team_code") == 3).sum().alias("nCT"))
    print(f"max per side per snapshot: T {side['nT'].max()}, CT {side['nCT'].max()}; "
          f"snapshots with at least one side above 5: {side.filter((pl.col('nT') > 5) | (pl.col('nCT') > 5)).height}")

    # splits and plants
    per_match = pl_.group_by("match_key").agg(pl.col("split").n_unique().alias("n"))
    print(f"matches with more than one split value: {per_match.filter(pl.col('n') > 1).height}; "
          f"matches in players table: {per_match.height}")
    print(f"rounds by split: {rd.group_by('split').len().sort('split').rows()}; "
          f"matches by split: {rd.group_by(['split']).agg(pl.col('match_key').n_unique()).sort('split').rows()}")
    bad_plant = rd.filter(pl.col("plant_tick").is_not_null() & (
        (pl.col("plant_tick") < pl.col("live_start_tick")) | (pl.col("plant_tick") >= pl.col("round_end_tick"))))
    print(f"plant ticks outside the live window: {bad_plant.height}; rounds with a plant: "
          f"{rd.filter(pl.col('plant_tick').is_not_null()).height} of {rd.height}")

    # score check
    sc = rd.filter(pl.col("ct_score") + pl.col("t_score") == pl.col("round") - 1)
    print(f"ct_score + t_score == round - 1: {sc.height} of {rd.height}")
    print(f"ct_win share (rounds): {rd['ct_win'].mean():.4f}; round_number == round for all rows: "
          f"{rd.filter(pl.col('round_number') == pl.col('round')).height == rd.height}")
    print(f"rounds with alive_gap_snapshots > 0: {rd.filter(pl.col('alive_gap_snapshots') > 0).height}; "
          f"total gap snapshots: {int(rd['alive_gap_snapshots'].sum())}")
    print(f"file sizes: players {P.stat().st_size / 1e6:.2f} MB, rounds {R.stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
