"""verify D9-12 step 1: schema and row probe of the raw tome channels used for the round_start check.

Read-only. Reads only parquet schemas and row counts (lazy scans) from the tomes.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
CHANNELS = ["round_start", "round_end", "round_state", "bomb_state", "player_spawn", "player_death", "round_mvp"]


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


for year, (base, suf) in TOMES.items():
    print(f"=== tome {year}: {base}")
    for ch in CHANNELS:
        fd = base / f"basic_{ch}{suf}"
        fs = files_of(fd)
        if not fs:
            print(f"  {ch}: NO FILES in {fd}")
            continue
        lf = pl.scan_parquet(fs)
        schema = lf.collect_schema()
        n = lf.select(pl.len()).collect().item()
        print(f"  {ch}: files={len(fs)} rows={n}")
        print("     cols:", ", ".join(f"{k}:{v}" for k, v in schema.items()))
