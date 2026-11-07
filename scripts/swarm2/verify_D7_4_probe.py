# Probe: schemas and file counts for the raw 2024 tome channels needed for the D7-4 recompute.
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
for ch in ["round_end", "bomb_state", "bomb_defuse", "bomb_action", "round_start", "round_state"]:
    fd = BASE / f"basic_{ch}{SUF}"
    fs = files(fd)
    print(f"== {ch}: files={len(fs)}")
    if fs:
        lf = pl.scan_parquet(fs)
        print("   schema:", {k: str(v) for k, v in lf.collect_schema().items()})
        print("   rows:", lf.select(pl.len()).collect().item())
hdr = files(BASE / "header")
print("header files:", len(hdr))
hs = pl.scan_parquet(hdr).collect_schema()
print("header cols:", list(hs.keys()))
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
print("2026 header files:", files(T26 / "header"))
print("2026 channels sample:", sorted([p.name for p in T26.iterdir()])[:45])
