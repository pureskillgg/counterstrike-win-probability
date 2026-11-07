"""D6-4 verifier: schema probe (read-only) for the 2024 tome channels used by the verification."""
import sys, glob
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
for ch in ["round_end", "round_state", "bomb_state", "player_death", "player_info", "player_spawn", "round_start", "bomb_action", "header"]:
    if ch == "header":
        fd = BASE / "header"
    else:
        cands = sorted(BASE.glob(f"basic_{ch}.*"))
        fd = cands[0] if cands else None
    if fd is None or not fd.exists():
        print(ch, "MISSING"); continue
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    sch = pl.read_parquet_schema(files[0])
    print(f"{ch}: files={len(files)} cols={list(sch.items())}")
