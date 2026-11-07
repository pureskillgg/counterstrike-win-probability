"""Shared loaders for swarm D5 (round-numbering hypothesis, 2024 tome).

Reads only the 2024 tome on G:. Never writes under E:. Column projection is applied
to every read so one job stays well under 3 GB.
"""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def channel_files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def read(ch, cols=None):
    fs = channel_files(ch)
    if not fs:
        raise FileNotFoundError(f"no dataframe files for channel {ch}")
    lf = pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")
    if cols:
        lf = lf.select(cols)
    return lf.collect()


def header():
    f = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0]
    return pl.read_parquet(f, columns=[
        "key", "match_id", "map_name", "match_date", "tick_rate",
        "t_starters_score_final", "ct_starters_score_final", "match_type"])


def day_of(match_key_col):
    return match_key_col.str.extract(r"csds/2024/02/(\d\d)/", 1)
