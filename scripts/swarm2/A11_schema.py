"""A11 schema probe (read-only): column names and dtypes for channels used in the EDA claim checks."""
import glob
from pathlib import Path
import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"

def folder(ch):
    return TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")

for ch in ["header", "round_end", "round_start", "round_state", "bomb_state", "rank_update",
           "weapon_fire", "weapon_action", "molotov_state", "grenade_state", "grenade_bounce", "molotov_fire"]:
    files = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    lf = pl.scan_parquet(files)
    sch = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"== {ch}: files={len(files)} rows={n}")
    print("   " + ", ".join(f"{k}:{v}" for k, v in sch.items()))
