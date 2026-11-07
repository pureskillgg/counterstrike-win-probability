"""verify D9-9, step 1: schema and header probe of the 2024-02 tome (read-only).

Prints column names and dtypes for the tables this verification uses, and a few header values
(build number fields, starter score fields). Reads only the first parquet file of each table.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def first_files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def show(name, fd):
    files = first_files(fd)
    print(f"== {name}: {len(files)} parquet files in {fd.name}")
    if not files:
        return
    lf = pl.scan_parquet(files[:1])
    sch = lf.collect_schema()
    for c, t in sch.items():
        print(f"   {c}: {t}")


show("header", BASE / "header")
for t in ["round_end", "round_start", "round_state", "bomb_state", "bomb_action", "bomb_defuse",
          "player_info", "player_death", "player_personal", "score_update", "team_change"]:
    fd = BASE / f"basic_{t}{SUF}"
    if fd.exists():
        show(t, fd)
    else:
        print(f"== {t}: folder missing")

hdr = pl.scan_parquet(first_files(BASE / "header")).select(
    [c for c in ["key", "build_num", "game_directory", "match_date", "map_name",
                 "t_starters_score_final", "ct_starters_score_final", "rushb_version"]
     if c in pl.scan_parquet(first_files(BASE / "header")).collect_schema().names()]
).head(5).collect()
print("\nheader sample:")
print(hdr)
