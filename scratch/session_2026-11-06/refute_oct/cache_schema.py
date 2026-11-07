import polars as pl, sys
sys.stdout.reconfigure(encoding="utf-8")
base = "G:/pureskill/csds-cs2/data/combined/"
for f in ["cache_snapshot_2026_oct23_31.parquet", "cache_snapshot_2026_ext.parquet", "cache_snapshot_2026.parquet", "cache_pos_2026.parquet"]:
    lf = pl.scan_parquet(base + f)
    names = lf.collect_schema().names()
    print(f, len(names), names[:40])
    print("rows", lf.select(pl.len()).collect().item(), flush=True)
