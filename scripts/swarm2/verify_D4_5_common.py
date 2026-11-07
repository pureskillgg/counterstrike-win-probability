"""Shared helpers for the D4-5 verification (verify_D4_5_*). Read-only. Column-projected lazy scans only.

ASSUMPTION (convention copied from scripts\\model\\bakeoff.py via the D4 scripts, re-checked here):
  swap_expected(r, mr) = (r == mr // 2) or (r > mr and (r - mr - 3) % 6 == 0)
  2024 tome has no max_rounds column in header, so mr = 24 (A12 per the claim).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
T, CT = 2, 3
DAY_RE = r"csds/\d{4}/\d\d/(\d\d)/"


def files(base, ch):
    out = []
    for d in sorted(base.glob(f"basic_{ch}.*")):
        out += sorted(glob.glob(str(d / "dataframe_*")))
    return out


def scan(base, ch, cols):
    return pl.scan_parquet(files(base, ch)).select(cols)


def header(base, cols):
    return pl.scan_parquet(sorted(glob.glob(str(base / "header" / "dataframe_*")))).select(cols).collect()


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def swap_expected_expr(mr_expr):
    r = pl.col("round")
    return (r == (mr_expr // 2)) | ((r > mr_expr) & (((r - mr_expr - 3) % 6) == 0))


def day_expr():
    return pl.col("match_key").str.extract(DAY_RE, 1).cast(pl.Int32)
