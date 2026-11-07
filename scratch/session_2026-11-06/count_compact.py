"""Read-only: for each path, scan ONLY match_key and print one line with total rows,
distinct matches, and rows/distinct matches inside csds/2026/10/26..2026/10/31. No other columns are read."""
import os
import sys
import polars as pl

LO, HI = "2026/10/26", "2026/10/31"
for path in sys.argv[1:]:
    try:
        df = pl.scan_parquet(path).select(pl.col("match_key")).with_columns(
            day=pl.col("match_key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1))
        tot = df.select(pl.len().alias("rows"), pl.col("day").null_count().alias("nomatch"),
                        pl.col("day").min().alias("d0"), pl.col("day").max().alias("d1")).collect().row(0, named=True)
        w = df.filter((pl.col("day") >= LO) & (pl.col("day") <= HI)).select(
            pl.len().alias("wrows"), pl.col("match_key").n_unique().alias("wmatches"),
            pl.col("day").n_unique().alias("wdays")).collect().row(0, named=True)
        print(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path),
              "rows", tot["rows"], "nomatch", tot["nomatch"], "span", tot["d0"], "to", tot["d1"],
              "| window rows", w["wrows"], "matches", w["wmatches"], "days", w["wdays"], flush=True)
    except Exception as e:  # keep going on bad files
        print("ERR", path, str(e)[:160], flush=True)
