"""Read-only schema and page-count peek for the flow channels and header.

Run: pixi run python -I scripts\\eda\\schema_peek.py
Reads parquet footers only (no data pages loaded beyond schema).
"""
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
CHANNELS = ["round_end", "round_start", "round_state", "bomb_state", "bomb_action",
            "bomb_defuse", "round_mvp", "score_update", "player_death", "player_info"]


def folder_for(ch):
    return TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")


def main():
    for ch in ["header"] + CHANNELS:
        files = sorted(glob.glob(str(folder_for(ch) / "dataframe_*")))
        schemas = {}
        total_rows = 0
        for f in files:
            sch = pl.read_parquet_schema(f)
            key = tuple((k, str(v)) for k, v in sch.items())
            schemas.setdefault(key, []).append(Path(f).name)
            total_rows += pl.scan_parquet(f).select(pl.len()).collect().item()
        print(f"== {ch}: pages={len(files)} rows={total_rows} distinct_schemas={len(schemas)}")
        for i, (key, names) in enumerate(schemas.items()):
            print(f"  schema {i} (pages={len(names)}, first={names[0]}):")
            for name, dtype in key:
                print(f"    {name}: {dtype}")
        print()


if __name__ == "__main__":
    main()
