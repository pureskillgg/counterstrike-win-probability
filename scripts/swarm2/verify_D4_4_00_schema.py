r"""Verifier D4-4: schema probe for the 2024 tome (read-only, lazy schema only).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_00_schema.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
CHANNELS = ["header", "round_end", "bomb_state", "round_state", "round_start", "player_info"]


def channel_dir(ch):
    if ch == "header":
        return BASE / "header"
    hits = sorted(BASE.glob(f"basic_{ch}.*"))
    return hits[0]


for ch in CHANNELS:
    fd = channel_dir(ch)
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    lf = pl.scan_parquet(files)
    schema = lf.collect_schema()
    print(f"== {ch}: files={len(files)} dir={fd.name}")
    for k, v in schema.items():
        print(f"   {k}: {v}")
    n = lf.select(pl.len()).collect().item()
    print(f"   rows={n}")
