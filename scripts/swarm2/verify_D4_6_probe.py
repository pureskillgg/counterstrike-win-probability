"""verify D4-6 schema probe: column names and dtypes for the channels used by the verifier (2024 and 2026 tomes).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D4_6_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
CHANNELS = ["round_state", "round_end", "player_death", "bomb_state", "header"]

for era, base in TOMES.items():
    for ch in CHANNELS:
        dirs = sorted(base.glob(f"basic_{ch}.*")) if ch != "header" else [base / "header"]
        if not dirs or not dirs[0].exists():
            print(era, ch, "MISSING")
            continue
        files = sorted(glob.glob(str(dirs[0] / "dataframe_*")))
        sch = pl.read_parquet_schema(files[0])
        print(f"{era} {ch}: dir={dirs[0].name} files={len(files)}")
        print("   ", {k: str(v) for k, v in sch.items()})
