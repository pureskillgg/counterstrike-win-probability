"""D3-4 verifier: schema probe for 2024 tome channels used by the swap claim (read-only, lazy scan)."""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
def folder(ch):
    return TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
for ch in ["header", "round_end", "round_state", "bomb_defuse", "bomb_action", "bomb_state"]:
    fs = sorted(glob.glob(str(folder(ch) / "dataframe_*")))
    print(f"== {ch}: {len(fs)} dataframe files")
    if not fs:
        continue
    lf = pl.scan_parquet(fs[0])
    print("  schema:", lf.collect_schema().items().__class__ and dict(lf.collect_schema()))
    print("  rows in first file:", lf.select(pl.len()).collect().item())
