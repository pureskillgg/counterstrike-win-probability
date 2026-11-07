"""D3-2 verifier probe 2: schemas of death/round channels and E: raw layout for one day (read only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ["player_death", "round_start", "player_info"]:
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    print(f"== {ch}: {len(fs)} files")
    if fs:
        print(pl.read_parquet_schema(fs[0]))
RAW = Path(r"E:\csds_final\csds\2024\02\12")
print("\nE raw 2024/02/12 exists:", RAW.exists())
if RAW.exists():
    kids = sorted(p.name for p in RAW.iterdir())
    print("match folders under 2024/02/12:", len(kids), kids[:3])
    if kids:
        p = RAW / kids[0]
        print("contents of first:", sorted(x.name for x in p.iterdir())[:40])
TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12")
print("E tail 2024/02/12 exists:", TAIL.exists())
if TAIL.exists():
    kids = sorted(p.name for p in TAIL.iterdir())
    print("tail match folders:", len(kids), kids[:3])
