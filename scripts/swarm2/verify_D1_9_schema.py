import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
S24 = ".2023-12-28,2024-02-25"
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
S26 = ".2026-09-18,2026-10-08"
for name, base, suf in [("2024", T24, S24), ("2026", T26, S26)]:
    for ch in ["round_end", "player_info", "header"]:
        if ch == "header":
            files = sorted(glob.glob(str(base / "header" / "dataframe_*")))
        else:
            files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
        print(name, ch, "files:", len(files))
        if files:
            print("  schema:", dict(pl.scan_parquet(files[0]).collect_schema()))
            n = pl.scan_parquet(files).select(pl.len()).collect().item()
            print("  rows:", n)
