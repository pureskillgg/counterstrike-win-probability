"""Verifier D2-4: schema probe (read-only). Prints column names and dtypes for the channels used."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}

for label in ["2024", "2026"]:
    for ch in ["round_state", "round_end", "bomb_state", "player_info", "player_death"]:
        fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
        files = sorted(glob.glob(fd + "/dataframe_*"))
        sch = pl.scan_parquet(files).collect_schema()
        n = pl.scan_parquet(files).select(pl.len()).collect().item()
        print(label, ch, "files:", len(files), "rows:", n)
        print("   ", [(k, str(v)) for k, v in sch.items()])
        if ch in ("round_state", "bomb_state"):
            ev = pl.scan_parquet(files).select(pl.col("event_type").unique()).collect()
            print("    event_types:", sorted(ev["event_type"].to_list()))
