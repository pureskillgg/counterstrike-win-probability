import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOTS = {
    "2024": Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"),
    "2026": Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"),
}
CH = ["round_state", "bomb_state", "player_death", "round_start", "score_update", "round_mvp", "header", "player_info"]
for era, base in ROOTS.items():
    for ch in CH:
        fds = sorted(base.glob(f"basic_{ch}.*")) if ch != "header" else [base / "header"]
        if not fds or not fds[0].exists():
            print(era, ch, "MISSING"); continue
        files = sorted(glob.glob(str(fds[0] / "dataframe_*")))
        if not files:
            print(era, ch, "no files in", fds[0]); continue
        sch = pl.read_parquet_schema(files[0])
        n = pl.scan_parquet(files).select(pl.len()).collect().item()
        print(f"{era} {ch} rows={n} :: " + ", ".join(f"{k}:{v}" for k, v in sch.items()))
