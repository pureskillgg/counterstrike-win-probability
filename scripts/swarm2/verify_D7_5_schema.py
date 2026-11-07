"""verify D7-5 step 0: schema and vocabulary probe for the 2024 tome (read-only, projection only).
Prints column schemas and small value vocabularies for the tables used by the verifier.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def schema(name, folder):
    fs = files(folder)
    print(f"\n### {name}: {len(fs)} files")
    print(dict(pl.read_parquet_schema(fs[0])))
    return fs


schema("header", BASE / "header")
for ch in ["round_end", "round_state", "bomb_state", "player_death"]:
    schema(ch, BASE / f"basic_{ch}{SUF}")

H = pl.scan_parquet(files(BASE / "header"))
print("\nheader sample:")
print(H.head(3).collect())

RS = pl.scan_parquet(files(BASE / f"basic_round_state{SUF}"))
print("\nround_state event_type counts:")
print(RS.group_by("event_type").len().sort("len", descending=True).collect())

BS = pl.scan_parquet(files(BASE / f"basic_bomb_state{SUF}"))
print("\nbomb_state event_type counts:")
print(BS.group_by("event_type").len().sort("len", descending=True).collect())

RE = pl.scan_parquet(files(BASE / f"basic_round_end{SUF}"))
print("\nround_end: winner_team_code x win_reason_code x win_reason_message (player_count==20):")
print(RE.filter(pl.col("player_count") == 20)
      .group_by(["winner_team_code", "win_reason_code", "win_reason_message"]).len()
      .sort(["winner_team_code", "win_reason_code"]).collect())
print("\nround_end player_count counts:")
print(RE.group_by("player_count").len().sort("len", descending=True).collect())

PD = pl.scan_parquet(files(BASE / f"basic_player_death{SUF}"))
pd_cols = pl.read_parquet_schema(files(BASE / f"basic_player_death{SUF}")[0])
print("\nplayer_death columns:", list(pd_cols.keys()))
