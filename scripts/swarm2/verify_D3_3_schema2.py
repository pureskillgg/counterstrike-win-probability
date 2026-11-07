"""Schema probe 2 for verify_D3_3: player-side channels and raw score fields (read-only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ["player_spawn", "player_death", "weapon_fire", "player_info", "round_mvp", "player_hurt", "bomb_action", "bomb_defuse", "player_personal"]:
    d = TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(d / "dataframe_*")))
    sch = pl.read_parquet_schema(files[0])
    print(f"== {ch}: files={len(files)} cols={list(sch.keys())}")
hd = TOME / "header"
print("header files:", sorted(glob.glob(str(hd / "dataframe_*"))))
