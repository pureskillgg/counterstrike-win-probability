"""verify D9-10 probe: schemas and row counts for the 2024-02 tome tables used by the claim.

Lazy scan only (schema + count), no full load. Read-only on the tome.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def fd_for(name):
    return BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"


for name in ["header", "round_end", "player_info", "player_death", "bomb_state", "round_start", "round_state"]:
    fs = files_of(fd_for(name))
    lf = pl.scan_parquet(fs)
    schema = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"== {name}: files={len(fs)} rows={n}")
    print("   ", {k: str(v) for k, v in schema.items()})
