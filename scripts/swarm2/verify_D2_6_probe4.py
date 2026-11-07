"""verify_D2_6 probe 4: explain the 2026 denominator (claim 23,210 vs recomputed 24,323). Reads D2 intermediate table only; prints only."""
import sys
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
t = pl.read_parquet(r"G:\pureskill\csds-cs2\logs\swarm2\D2_truth_2026.parquet")
print("D2_truth_2026 rows:", t.height)
print("sx null:", t.filter(pl.col("sx").is_null()).height, " t0 null:", t.filter(pl.col("t0").is_null()).height,
      " ct0 null:", t.filter(pl.col("ct0").is_null()).height, " R null:", t.filter(pl.col("R").is_null()).height)
print("rows with sx and t0 and ct0 non-null:", t.filter(pl.col("sx").is_not_null() & pl.col("t0").is_not_null() & pl.col("ct0").is_not_null()).height)
print("rows with sx non-null:", t.filter(pl.col("sx").is_not_null()).height)
print("rows with t0/ct0 non-null but sx null:", t.filter(pl.col("sx").is_null() & pl.col("t0").is_not_null()).height)
print("columns:", t.columns)
