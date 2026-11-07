"""A13: verify feature_spec.md section 3 counts against the 2026 tome (read-only, stdout only).

Checks: round_end winner codes; win_reason -> winner map counts; bomb_planted planter side;
bomb_defused planter side; distinct round_state event types; round_freeze_end rows.
Output: logs/swarm2/A13_spec_counts.log (captured by the caller).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes/2026-10-01_to_08/tome/csds"
SUFFIX = ".2026-09-18,2026-10-08"


def collections_counter(s):
    return list(collections.Counter(s.to_list()).items())


def read(channel, cols):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = read("round_end", ["match_key", "round", "winner_team_code", "win_reason_code"])
print(f"round_end rows: {RE.height}")
print(f"winner code counts: {dict(sorted(collections_counter(RE['winner_team_code'])))}")
wr = RE.group_by(["win_reason_code", "winner_team_code"]).agg(pl.len().alias("n")).sort(["win_reason_code"])
print("win_reason_code -> winner_team_code counts:")
for r in wr.iter_rows(named=True):
    print(f"  reason {r['win_reason_code']} -> winner {r['winner_team_code']}: {r['n']}")

BS = read("bomb_state", ["match_key", "round", "event_type", "player_team_code"])
pl_rows = BS.filter(pl.col("event_type") == "bomb_planted")
print(f"\nbomb_planted rows: {pl_rows.height}; planter code counts: "
      f"{dict(sorted(collections_counter(pl_rows['player_team_code']), key=lambda x: str(x[0])))}")
dfs = BS.filter(pl.col("event_type") == "bomb_defused")
print(f"bomb_defused rows: {dfs.height}; planter code counts: "
      f"{dict(sorted(collections_counter(dfs['player_team_code']), key=lambda x: str(x[0])))}")
print(f"bomb event types: {sorted(BS['event_type'].unique().to_list())}")

RS = read("round_state", ["match_key", "round", "event_type"])
print(f"\nround_state rows: {RS.height}; distinct event_type: {RS['event_type'].n_unique()}")
fz = RS.filter(pl.col("event_type") == "round_freeze_end")
print(f"round_freeze_end rows: {fz.height}; distinct (match, round): "
      f"{fz.select(['match_key', 'round']).unique().height}")
rs_counts = RS.group_by("event_type").agg(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches")).sort("rows", descending=True)
for r in rs_counts.iter_rows(named=True):
    print(f"  {r['event_type']:<34}{r['rows']:>8}{r['matches']:>7}")
