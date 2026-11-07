"""verify D3-3 probe: schemas, header max_rounds mix, sample rows (2024 G: tome). Read-only."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


for ch in ["round_end", "round_state", "bomb_action", "bomb_defuse", "bomb_state", "round_start"]:
    lf = scan(ch)
    print(f"== {ch} schema:")
    print(lf.collect_schema())
    print(lf.head(3).collect())
    print()

hdr_files = sorted(glob.glob(str(TOME / "header" / "dataframe_*")))
hdr = pl.read_parquet(hdr_files)
print("== header schema:")
print(hdr.schema)
print("header rows:", hdr.height)
print("max_rounds dist:", hdr.group_by("max_rounds").len().sort("max_rounds").rows())
print("match_date sample:", hdr.select(["key", "match_date", "max_rounds"]).head(3).rows())
