"""D2 schema probe: column names and dtypes for the channels used by hypothesis D2."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
CH = ["round_end", "round_start", "round_state", "player_info", "player_spawn",
      "bomb_state", "bomb_action", "bomb_defuse", "team_change", "score_update", "player_death"]


def folder(base, ch):
    hits = sorted(glob.glob(str(base / f"basic_{ch}.*")))
    return Path(hits[0]) if hits else None


for label, base in [("2024", T24), ("2026", T26)]:
    for ch in CH:
        fd = folder(base, ch)
        if fd is None:
            print(f"[{label}] {ch}: MISSING")
            continue
        files = sorted(glob.glob(str(fd / "dataframe_*")))
        sch = pl.read_parquet_schema(files[0]) if files else {}
        nrows = pl.scan_parquet(files).select(pl.len()).collect().item() if files else 0
        print(f"[{label}] {ch}: files={len(files)} rows={nrows}")
        for k, v in sch.items():
            print(f"    {k}: {v}")
