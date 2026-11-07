"""verify D7-3: schema and build_num survey (read-only). Writes nothing except stdout."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"


def first_file(folder):
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    return fs[0] if fs else None


for label, base, suf in [("2024", T24, SUF24), ("2026", T26, None)]:
    print(f"=== {label} ===")
    hdr = first_file(base / "header")
    print("header file:", hdr)
    print("header schema:", pl.read_parquet_schema(hdr))
    for ch in ["round_end", "bomb_state", "round_start", "round_state", "bomb_defuse", "bomb_action", "player_spawn"]:
        cands = sorted(glob.glob(str(base / f"basic_{ch}*")))
        cands = [c for c in cands if Path(c).is_dir()]
        if not cands:
            print(f"  {ch}: no folder")
            continue
        f = first_file(Path(cands[0]))
        if f is None:
            print(f"  {ch}: no dataframe file in {cands[0]}")
            continue
        print(f"  {ch} ({Path(cands[0]).name}) schema:", dict(pl.read_parquet_schema(f)))
