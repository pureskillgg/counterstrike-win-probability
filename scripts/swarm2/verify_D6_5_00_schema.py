"""D6-5 verifier step 0: schema and size probe for the 2024 tome channels used by the claim."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def files(ch):
    if ch == "header":
        return sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


for ch in ["header", "round_end", "round_state", "bomb_state", "player_info", "player_death", "player_spawn"]:
    fs = files(ch)
    size = sum(Path(f).stat().st_size for f in fs)
    schema = pl.read_parquet_schema(fs[0]) if fs else {}
    print(f"{ch}: files={len(fs)} total_MB={size / 1e6:.1f}")
    print("   schema:", dict(schema))
