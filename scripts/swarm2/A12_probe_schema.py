"""A12 probe: schemas and row counts of the 2026 tome channels used by the weapon and grenade audit."""
import glob
from pathlib import Path

import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"


def folder(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected one folder, got {[h.name for h in hits]}")
    return hits[0]


for ch in ["player_death", "weapon_fire", "grenade_state", "grenade_bounce", "round_end", "item_pickup"]:
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    lf = pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")
    sch = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"== {ch}: files={len(fs)} rows={n:,}")
    print("   " + ", ".join(f"{k}:{v}" for k, v in sch.items()))
