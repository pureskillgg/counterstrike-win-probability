"""verify D5-6, step 1: schema and vocabulary probe of the 2024 tome (independent of D5 scratch files).

Reads only the 2024 tome on G: via lazy scans with column projection. Prints schemas and the
round_end reason vocabulary, and the round_state / bomb_state event_type counts.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(80)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    print(f"[{ch}] files: {len(fs)}")
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


for ch in ["round_end", "round_state", "bomb_state", "player_death", "round_start"]:
    lf = scan(ch)
    print(f"  schema[{ch}]:", {k: str(v) for k, v in lf.collect_schema().items()})

RE = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                               "win_reason_message", "legacy_code", "player_count"]).collect()
print("\nround_end rows:", RE.height)
print("winner_team_code:", RE.group_by("winner_team_code").len().sort("winner_team_code").to_dicts())
print("player_count:", RE.group_by("player_count").len().sort("player_count").to_dicts())
print("reason vocabulary (code, message, winner_team_code, n):")
print(RE.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len()
      .sort(["win_reason_code", "win_reason_message", "winner_team_code"]))
print("legacy_code x winner:")
print(RE.group_by(["legacy_code", "winner_team_code"]).len().sort(["legacy_code", "winner_team_code"]))

BS = scan("bomb_state").select(["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect()
print("\nbomb_state bomb rows by event_type:", BS.group_by("event_type").len().to_dicts())

RS = scan("round_state").select(["event_type"]).group_by("event_type").len().collect()
print("\nround_state event_type counts (top 40):")
print(RS.sort("len", descending=True).head(40))
