"""verify D5-9 probe: schemas and event-type inventory for the 2024 tome (lazy, projected).

Reads only G:\\pureskill\\csds-cs2\\tomes\\2024-02-g\\tome\\csds. Never writes under E:.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


for ch in ["round_state", "round_end", "bomb_state", "player_death", "round_start"]:
    fs = files(ch)
    print(f"== {ch}: {len(fs)} files")
    if not fs:
        continue
    lf = pl.scan_parquet(fs[0])
    print(lf.collect_schema())

hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("== header files:", len(hf))
print(pl.scan_parquet(hf[0]).collect_schema())
