"""verify D4-5, step 0: schema and row-count probe for the 2024 tome channels used by the check.

Reads only the columns needed. Writes to stdout only (captured into logs\\swarm2\\verify_D4_5.log by the runner).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D4_5_00_schema.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def dirs(ch):
    return sorted(BASE.glob(f"basic_{ch}.*"))


def files(ch):
    out = []
    for d in dirs(ch):
        out += sorted(glob.glob(str(d / "dataframe_*")))
    return out


print("polars", pl.__version__, "python", sys.version.split()[0])
for ch in ["round_state", "round_end", "bomb_state", "player_death", "other_death", "round_start", "header"]:
    fs = files(ch) if ch != "header" else sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
    if not fs:
        print(f"\n{ch}: no dataframe files found")
        continue
    lf = pl.scan_parquet(fs)
    schema = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print(f"\n{ch}: files={len(fs)} rows={n}")
    for k, v in schema.items():
        print(f"   {k}: {v}")

# event types present in round_state and bomb_state
for ch in ["round_state", "bomb_state"]:
    fs = files(ch)
    ev = (pl.scan_parquet(fs).select(pl.col("event_type")).group_by("event_type").agg(pl.len().alias("n"))
          .sort("n", descending=True).collect())
    print(f"\n{ch} event_type counts:")
    print(ev)

# header columns
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = pl.scan_parquet(hf).collect_schema()
print("\nheader schema:", dict(H))
