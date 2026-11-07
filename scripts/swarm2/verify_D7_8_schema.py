"""Schema probe for D7-8 verification (2024 tome, read-only)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
for ch in ["header", "round_state", "round_end", "round_start", "bomb_state", "player_spawn", "player_info", "player_personal", "player_death", "round_mvp"]:
    fd = BASE / (ch if ch == "header" else f"basic_{ch}{SUF}")
    fs = files(fd)
    if not fs:
        print(ch, "NO FILES"); continue
    sch = pl.read_parquet_schema(fs[0])
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in fs[:1])
    print(f"== {ch}: files={len(fs)} schema={dict(sch)}")
    print("   first-file rows:", n)
    print(pl.read_parquet(fs[0], n_rows=3))
