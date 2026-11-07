"""verify D7-6 step a: schema and row counts of the raw 2024 tome tables used by the verifier.
Reads schemas only (lazy scan), plus row counts per table via lazy count. Stdout only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
TABLES = ["round_end", "round_state", "bomb_state", "round_start", "player_spawn", "player_death",
          "player_info", "player_personal", "round_mvp", "bomb_action", "bomb_defuse", "other_death"]


def files(name):
    return sorted(glob.glob(str(BASE / f"basic_{name}{SUF}" / "dataframe_*")))


for t in TABLES:
    fs = files(t)
    if not fs:
        print(f"{t}: NO FILES")
        continue
    lf = pl.scan_parquet(fs)
    sch = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"{t}: files={len(fs)} rows={n}")
    print("   ", {k: str(v) for k, v in sch.items()})

h = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name"])
print("header rows:", h.height, "| build_num:", h.group_by("build_num").len().sort("build_num").rows())
print("header sample:", h.head(3).rows())
