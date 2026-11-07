import polars as pl, sys
for p in sys.argv[1:]:
    lf = pl.scan_parquet(p)
    print("==", p)
    print(lf.collect_schema().names()[:60])
    print("rows", lf.select(pl.len()).collect().item())
