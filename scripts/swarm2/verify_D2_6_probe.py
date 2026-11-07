"""Probe for verify_D2_6: channel folders, event types, duplicate keys. Prints only (writes no files).

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def folders(label, ch):
    return sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))


for label in ["2024", "2026"]:
    for ch in ["round_state", "round_end", "player_info", "player_death", "bomb_state"]:
        fs = folders(label, ch)
        print(label, ch, "folders:", len(fs), [Path(f).name for f in fs])
        if fs:
            files = sorted(glob.glob(fs[0] + "/dataframe_*"))
            print("   parquet files:", len(files), Path(files[0]).name if files else None)

for label in ["2024", "2026"]:
    fs = folders(label, "round_state")
    rs = pl.scan_parquet(sorted(glob.glob(fs[0] + "/dataframe_*"))).select(["event_type"]).collect()
    print(label, "round_state event_type counts:", rs.group_by("event_type").len().sort("event_type").rows())
    fs = folders(label, "bomb_state")
    bs = pl.scan_parquet(sorted(glob.glob(fs[0] + "/dataframe_*"))).select(["event_type"]).collect()
    print(label, "bomb_state event_type counts:", bs.group_by("event_type").len().sort("event_type").rows())
    fs = folders(label, "round_end")
    re_ = pl.scan_parquet(sorted(glob.glob(fs[0] + "/dataframe_*"))).select(
        ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
    print(label, "round_end rows:", re_.height, "player_count counts:",
          re_.group_by("player_count").len().sort("len", descending=True).head(6).rows())
    dup = re_.filter(pl.col("player_count").is_in([20, 10])).group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print(label, "duplicate (match,round) among PC 20/10 rows:", dup.height)
    if dup.height:
        print("   sample dup keys:", dup.head(3).rows())
    print(label, "winner_team_code values:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
