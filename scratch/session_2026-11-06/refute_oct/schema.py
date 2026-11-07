import polars as pl, sys
base = "G:/pureskill/csds-cs2/data/"
for f in ["positions_1hz/2026-10-21.parquet", "status_1hz/2026-10-21.parquet", "combined/cache_snapshot_2026_oct23_31.parquet", "combined/cache_snapshot_2026_ext.parquet", "combined/cache_snapshot_2026.parquet"]:
    lf = pl.scan_parquet(base + f)
    print(f, lf.collect_schema().names())
    print(lf.head(3).collect())
    print("rows", lf.select(pl.len()).collect().item())
