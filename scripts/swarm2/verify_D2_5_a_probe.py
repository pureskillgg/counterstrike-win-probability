"""verify D2-5 probe: schema and reason-field value sets for round_end in both tomes.

Reads only round_end and bomb_state with column projection. Prints value sets so the
reason-code mapping used by the claim can be checked against the message strings.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_a_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def folder(label, ch):
    return sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]


def files(label, ch):
    return sorted(glob.glob(folder(label, ch) + "/dataframe_*"))


for label in ["2024", "2026"]:
    print(f"\n######## {label} ########")
    print("round_end folder:", folder(label, "round_end"))
    print("round_end files:", len(files(label, "round_end")))
    print("round_end schema:", pl.read_parquet_schema(files(label, "round_end")[0]))
    re_ = pl.scan_parquet(files(label, "round_end")).select(
        ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
         "legacy_code", "player_count"]).collect()
    print("round_end rows:", re_.height)
    print("player_count dist:")
    print(re_.group_by("player_count").len().sort("player_count"))
    print("win_reason_code x win_reason_message (all rows):")
    print(re_.group_by(["win_reason_code", "win_reason_message"]).len().sort("win_reason_code"))
    print("legacy_code dist (top):")
    print(re_.group_by("legacy_code").len().sort("len", descending=True).head(30))
    print("winner_team_code dist:")
    print(re_.group_by("winner_team_code").len().sort("winner_team_code"))
