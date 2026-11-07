"""Verifier D2-2 schema probe (2024 tome). Prints column names and small samples only.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_2_schema.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(12)
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
CHANNELS = ["round_end", "round_state", "bomb_state", "player_info", "player_death", "header", "round_start"]


def ch_files(ch):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


for ch in CHANNELS:
    files = ch_files(ch) if ch != "header" else sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
    schema = pl.read_parquet_schema(files[0])
    print(f"\n== {ch}: {len(files)} files")
    print(dict(schema))
    if ch == "header":
        df = pl.scan_parquet(files).head(3).collect()
        print(df)
