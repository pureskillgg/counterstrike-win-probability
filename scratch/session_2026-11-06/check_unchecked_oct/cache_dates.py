import polars as pl, sys
for p in sys.argv[1:]:
    lf = pl.scan_parquet(p).select(["match_key","match_date"])
    m = lf.group_by("match_key").agg(pl.col("match_date").first()).collect()
    d = m["match_date"]
    print("==", p.split("/")[-1], "matches", m.height, "dtype", d.dtype)
    print(" min", d.min(), "max", d.max())
    s = m.with_columns(pl.col("match_date").cast(pl.Utf8).str.slice(0,10).alias("d"))
    s = s.with_columns(pl.col("d").str.to_date(strict=False).alias("dd"))
    print(" null dates", s["dd"].null_count())
    import datetime as dt
    def cnt(a,b):
        return s.filter((pl.col("dd")>=dt.date(*a))&(pl.col("dd")<=dt.date(*b))).height
    print(" Oct21-25 matches", cnt((2026,10,21),(2026,10,25)), "Oct14-20", cnt((2026,10,14),(2026,10,20)), "Oct26-31", cnt((2026,10,26),(2026,10,31)), "<=Oct13", cnt((2026,1,1),(2026,10,13)))
    print(" Oct 21 only", cnt((2026,10,21),(2026,10,21)))
