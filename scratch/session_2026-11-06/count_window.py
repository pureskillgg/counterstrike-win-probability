"""Read-only: scan ONLY the match_key column of a parquet file and count
rows / distinct matches per day for the csds/YYYY/MM/DD prefix. Window 2026/10/26-2026/10/31."""
import sys
import polars as pl

path = sys.argv[1]
lf = pl.scan_parquet(path).select(pl.col("match_key"))
df = lf.with_columns(
    day=pl.col("match_key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1)
)

overall = df.select(
    pl.len().alias("rows"),
    pl.col("match_key").null_count().alias("null_match_key"),
    pl.col("day").null_count().alias("no_regex_match"),
    pl.col("match_key").n_unique().alias("distinct_match_keys"),
    pl.col("day").n_unique().alias("distinct_days"),
    pl.col("day").min().alias("min_day"),
    pl.col("day").max().alias("max_day"),
).collect()
print("OVERALL", overall.row(0, named=True))

w = df.filter((pl.col("day") >= "2026/10/26") & (pl.col("day") <= "2026/10/31"))
wsum = w.select(
    pl.len().alias("rows"),
    pl.col("match_key").n_unique().alias("distinct_matches"),
    pl.col("day").n_unique().alias("days_present"),
).collect()
print("WINDOW_10_26_31", wsum.row(0, named=True))

per_day = (
    w.group_by("day")
    .agg(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches"))
    .sort("day")
    .collect()
)
print("PER_DAY_WINDOW", per_day.rows())

# day coverage across the file (days + match counts) to see the full span
cov = (
    df.group_by("day")
    .agg(pl.col("match_key").n_unique().alias("matches"))
    .sort("day")
    .collect()
)
print("DAY_COVERAGE_COUNT", cov.height, "first", cov["day"][0], "last", cov["day"][-1])
print("DAYS_WITH_MATCHES_ALL", cov.rows())
