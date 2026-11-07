"""verify D2-5: schema probe for bomb_state, player_death, round_state, player_info, round_end (both tomes)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
CH = ["bomb_state", "bomb_action", "bomb_defuse", "player_death", "round_state", "player_info", "round_end"]
for label in ["2024", "2026"]:
    for ch in CH:
        cands = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
        if not cands:
            print(label, ch, "MISSING")
            continue
        fs = sorted(glob.glob(cands[0] + "/dataframe_*"))
        sch = pl.read_parquet_schema(fs[0])
        print(label, ch, "files", len(fs), dict(sch))
