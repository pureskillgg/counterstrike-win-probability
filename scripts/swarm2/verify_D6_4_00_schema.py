"""D6-4 verifier: schema probe (reads parquet metadata only, no data load)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
for ch in ["round_end", "round_state", "bomb_state", "bomb_action", "bomb_defuse", "player_death", "player_info", "round_start", "player_spawn", "other_death"]:
    fd = BASE / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    sch = pl.read_parquet_schema(files[0])
    n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    print(f"{ch}: files={len(files)} rows={n}")
    for k, v in sch.items():
        print(f"    {k}: {v}")
hdr = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
print("header files", len(hdr), list(pl.read_parquet_schema(hdr[0]).keys()))
