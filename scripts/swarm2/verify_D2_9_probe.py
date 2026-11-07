"""verify D2-9 probe: schema, row counts and per-round event multiplicity for round_start / round_end /
round_state in both tomes. Lazy scans with projection only. Writes to stdout (captured into the log).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def channel_files(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    parts = sorted(glob.glob(fd[0] + "/dataframe_*"))
    return parts


def main():
    print("polars", pl.__version__)
    for label in ["2024", "2026"]:
        for ch in ["round_start", "round_end", "round_state"]:
            parts = channel_files(label, ch)
            lf = pl.scan_parquet(parts)
            schema = lf.collect_schema()
            n = lf.select(pl.len()).collect().item()
            print(f"\n== {label} basic_{ch}: parts={len(parts)} rows={n:,}")
            print("   schema:", dict(schema))
    print()


if __name__ == "__main__":
    main()
