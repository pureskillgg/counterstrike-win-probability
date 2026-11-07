"""Schema probe for W1 verification. Reads parquet schemas and a few rows only."""
import glob
import os
import sys

import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CHANNELS = [
    "basic_round_end",
    "basic_round_state",
    "basic_player_death",
    "basic_weapon_fire",
    "basic_player_hurt",
    "basic_player_info",
]

for ch in CHANNELS:
    folder = os.path.join(TOME, f"{ch}.2026-09-18,2026-10-08")
    files = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    print(f"== {ch}: {len(files)} files")
    lf = pl.scan_parquet(files)
    schema = lf.collect_schema()
    for name, dtype in schema.items():
        print(f"   {name}: {dtype}")
    n = lf.select(pl.len()).collect().item()
    print(f"   rows: {n}")
    print(lf.head(2).collect())
    sys.stdout.flush()

for p in [
    r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet",
    r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet",
    r"G:\pureskill\csds-cs2\data\state_sample\round_buy_types.parquet",
]:
    lf = pl.scan_parquet(p)
    print(f"== {os.path.basename(p)}")
    print(lf.collect_schema())
    print(lf.select(pl.len()).collect().item(), "rows")
    sys.stdout.flush()
