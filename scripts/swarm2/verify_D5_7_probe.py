"""D5-7 verifier, step 0: schema and event-type probe of the 2024 tome channels.

Reads only schemas and small aggregates on G: (no writes). Column projection on every read.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(80)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


for ch in ["round_state", "round_start", "round_end", "bomb_state", "other_death", "round_mvp"]:
    fs = files(ch)
    print(f"\n== {ch}: {len(fs)} dataframe files ==")
    if not fs:
        continue
    sch = pl.read_parquet_schema(fs[0])
    print("columns:", list(sch.keys()))
    lf = pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")
    print("rows:", lf.select(pl.len()).collect().item())
    if "event_type" in sch:
        print(lf.group_by("event_type").agg(pl.len().alias("rows")).sort("event_type").collect())
    if "label" in sch:
        print(lf.group_by("label").agg(pl.len().alias("rows")).sort("label").collect().head(60))

print("\n== header ==")
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("files:", len(hf))
print(pl.read_parquet_schema(hf[0]))
