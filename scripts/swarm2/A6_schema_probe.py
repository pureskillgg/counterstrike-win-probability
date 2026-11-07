import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOTS = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds") / "basic_round_end.2023-12-28,2024-02-25",
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds") / "basic_round_end.2026-09-18,2026-10-08",
}
for era, fd in ROOTS.items():
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    print(era, "files:", len(files), files[:2])
    sch = pl.read_parquet_schema(files[0])
    for k, v in sch.items():
        print("   ", k, v)
    lf = pl.scan_parquet(files)
    print("   rows:", lf.select(pl.len()).collect().item())
