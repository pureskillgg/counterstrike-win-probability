"""Read-only: for each per-map pos24 table, confirm the trainer's join requirements using
schema plus the key columns only (match_key, round, second): p0..p23 present, row count vs unique keys."""
import os
import sys
import polars as pl

need = ["match_key", "round", "second"] + [f"p{i}" for i in range(24)]
for path in sys.argv[1:]:
    lf = pl.scan_parquet(path)
    names = set(lf.collect_schema().names())
    missing = [c for c in need if c not in names]
    keys = lf.select(["match_key", "round", "second"]).select(
        pl.len().alias("rows"),
        pl.struct(["match_key", "round", "second"]).n_unique().alias("uniq_keys"),
    ).collect().row(0, named=True)
    print(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path),
          "missing_cols", missing if missing else "none",
          "rows", keys["rows"], "unique_keys", keys["uniq_keys"],
          "dup_keys" if keys["rows"] != keys["uniq_keys"] else "no_dups", flush=True)
