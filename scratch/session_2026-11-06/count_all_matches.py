"""Count distinct matches in each combined parquet (match identifiers only; no per-second rows read beyond the key column)."""
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2/data/combined")
for f in sorted(ROOT.glob("*.parquet")):
    try:
        lf = pl.scan_parquet(f)
        cols = lf.collect_schema().names()
        key = "match_key" if "match_key" in cols else next((c for c in cols if "match" in c.lower()), None)
        if key is None:
            print(f"{f.name}: no match column; columns start with {cols[:6]}")
            continue
        n = lf.select(pl.col(key).n_unique()).collect().item()
        print(f"{f.name}: {n:,} distinct {key}")
    except Exception as e:  # noqa: BLE001
        print(f"{f.name}: error {type(e).__name__}: {str(e)[:120]}")
