"""verify D2-6: schema probe for header and key channels (lazy schema only, no data pulled)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def fold(label, ch):
    if ch == "header":
        return TOME[label] / "header"
    return Path(sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0])


for label in ["2024", "2026"]:
    print(f"==== {label} header ====")
    h = fold(label, "header")
    files = sorted(glob.glob(str(h / "dataframe_*")))
    lf = pl.scan_parquet(files)
    print("files:", files)
    print("schema:")
    for k, v in lf.collect_schema().items():
        print("   ", k, v)
    print("rows:", lf.select(pl.len()).collect().item())
    print("head 3 (selected cols):")
    print(lf.head(3).collect())
