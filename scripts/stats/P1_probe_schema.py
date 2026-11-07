"""P1 probe: print schemas and sample rows for the channels used by the pistol-round stats.

Read-only. Lazy scans with a row limit; no full loads.
"""
import glob
import os

import polars as pl

ROOT = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CHANNELS = [
    "basic_round_end",
    "basic_round_state",
    "basic_round_start",
    "basic_weapon_fire",
    "basic_player_death",
    "basic_player_hurt",
    "basic_player_info",
]

for ch in CHANNELS:
    folders = glob.glob(os.path.join(ROOT, ch + ".*"))
    if not folders:
        print("MISSING", ch)
        continue
    folder = folders[0]
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    print("=" * 100)
    print(ch, "files:", [os.path.basename(f) for f in files])
    lf = pl.scan_parquet(files)
    schema = lf.collect_schema()
    for name, dtype in schema.items():
        print("   ", name, dtype)
    print("  head(3):")
    with pl.Config(tbl_cols=30, tbl_width_chars=250, fmt_str_lengths=60):
        print(lf.head(3).collect())
    n = lf.select(pl.len()).collect().item()
    print("  rows:", n)
