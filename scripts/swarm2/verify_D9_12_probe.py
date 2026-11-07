"""D9-12 verifier, step 0: schema and row-count probe for the round-related channels in both tomes.

Read-only. Uses polars lazy scans and reads only the schema plus a row count per channel.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
CHANNELS = ["round_start", "round_end", "round_state", "bomb_state", "bomb_action", "bomb_defuse"]

for tome, (base, suf) in TOMES.items():
    print(f"== tome {tome}")
    for ch in CHANNELS:
        fd = base / f"basic_{ch}{suf}"
        files = sorted(glob.glob(str(fd / "dataframe_*")))
        if not files:
            print(f"   {ch}: NO FILES in {fd}")
            continue
        lf = pl.scan_parquet(files)
        schema = lf.collect_schema()
        n = lf.select(pl.len()).collect().item()
        print(f"   {ch}: files={len(files)} rows={n} cols={list(schema.items())}")
