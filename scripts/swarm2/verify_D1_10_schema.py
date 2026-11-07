"""D1-10 verifier: schema probe for the 2026 tome tables used by the score-vs-winner check.
Reads parquet metadata only (no row data). Prints column names and dtypes.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
CHANNELS = ["round_end", "round_state", "score_update", "bomb_defuse", "bomb_action", "bomb_state", "round_start"]


def files(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


for ch in CHANNELS:
    fs = files(ch)
    print(f"== {ch}: {len(fs)} file(s)")
    if not fs:
        continue
    lf = pl.scan_parquet(fs)
    schema = lf.collect_schema()
    print("  columns:", ", ".join(f"{k}:{v}" for k, v in schema.items()))
    n = lf.select(pl.len()).collect().item()
    print("  rows:", n)
