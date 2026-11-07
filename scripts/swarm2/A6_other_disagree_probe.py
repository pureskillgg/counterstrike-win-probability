"""A6 probe: where are the non-13/25 disagreements (round_end winner vs score-derived or bomb side) located?
Reads the A6_threeway_2024.parquet written by A6_threeway_probe.py (scratchpad)."""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
P2 = pl.read_parquet(Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / "A6_threeway_2024.parquet")
oth = P2.filter(~pl.col("round").is_in([13, 25]) & pl.col("S_ns").is_not_null())
dis = oth.filter(pl.col("W") != pl.col("S_ns"))
print("non-13/25 scored rows:", oth.height, " disagree W vs S_ns:", dis.height)
print("disagreements by round (top 15):", dis.group_by("round").len().sort("len", descending=True).head(15).rows())
print("disagreements by day group:", dis.with_columns(
    pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01_07")).otherwise(pl.lit("d12_25")).alias("g")
).group_by("g").len().rows())
print("disagreements by code:", dis.group_by("code").len().sort("code").rows())
ob = P2.filter(~pl.col("round").is_in([13, 25]) & pl.col("B").is_not_null())
print("non-13/25 bomb rows:", ob.height, " W != B:", ob.filter(pl.col("W") != pl.col("B")).height)
print("W != B by round:", ob.filter(pl.col("W") != pl.col("B")).group_by("round").len().sort("round").rows())
print("W != B by day:", ob.filter(pl.col("W") != pl.col("B")).group_by("day").len().sort("day").rows())
