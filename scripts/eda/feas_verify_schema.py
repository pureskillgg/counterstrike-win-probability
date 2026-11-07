"""Scratch schema probe for the grenade verifier (read-only). Prints column names and dtypes."""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"

CHANNELS = ["grenade_state", "molotov_state", "molotov_fire", "player_hurt", "player_blind",
            "bomb_state", "bomb_action", "bomb_defuse", "round_end", "round_state"]


def folder_for(ch):
    hits = sorted(p for p in TOME.iterdir() if p.is_dir() and p.name.startswith(f"basic_{ch}."))
    if len(hits) != 1:
        raise SystemExit(f"{ch}: expected 1 folder, found {[h.name for h in hits]}")
    return hits[0]


for ch in CHANNELS:
    f = folder_for(ch)
    files = sorted(glob.glob(str(f / "dataframe_*")))
    lf = pl.scan_parquet(files)
    schema = lf.collect_schema()
    print(f"== {ch}: folder={f.name} files={len(files)} cols={len(schema)}")
    print("   " + ", ".join(f"{k}:{v}" for k, v in schema.items()))

rounds = pl.read_parquet(ROOT / "data" / "state_sample" / "rounds.parquet")
print("== rounds.parquet:", rounds.height, "rows")
print("   " + ", ".join(f"{k}:{v}" for k, v in rounds.schema.items()))
pos = pl.read_parquet(ROOT / "data" / "state_sample" / "positions_snapshots.parquet")
print("== positions_snapshots.parquet:", pos.height, "rows")
print("   " + ", ".join(f"{k}:{v}" for k, v in pos.schema.items()))
