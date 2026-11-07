"""verify D5-9, step 0: schema probe for the 2024 tome channels used by the check.

Reads parquet footers / one row only. Reads G: only. No writes.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_00_schema.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"

for ch in ["round_end", "round_state", "round_start", "player_death", "bomb_state", "player_info"]:
    folder = BASE / f"basic_{ch}{SUF}"
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        print(ch, "NO FILES")
        continue
    lf = pl.scan_parquet(fs[0])
    sch = lf.collect_schema()
    print(f"== {ch}: files={len(fs)} columns={len(sch)}")
    for name, dt in sch.items():
        print(f"   {name}: {dt}")
    head = lf.head(1).collect()
    print("   sample row:", {k: v for k, v in head.row(0, named=True).items() if k in list(sch.keys())[:12]})

hdr = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("== header files:", hdr)
hsch = pl.scan_parquet(hdr[0]).collect_schema()
print("   header columns:", list(hsch.keys()))
