"""P1 probe: shape and code distributions of the 1 Hz freeze-end sample and rounds.parquet.

Read-only. Prints compact summaries only.
"""
import polars as pl

SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"
ROUNDS = r"G:\pureskill\csds-cs2\data\state_sample\rounds.parquet"
BUYS = r"G:\pureskill\csds-cs2\data\state_sample\round_buy_types.parquet"

lf = pl.scan_parquet(SAMPLE)
print("SAMPLE schema:")
for k, v in lf.collect_schema().items():
    print("  ", k, v)
n_rows = lf.select(pl.len()).collect().item()
n_matches = lf.select(pl.col("match_key").n_unique()).collect().item()
print("rows", n_rows, "matches", n_matches)

sec = (lf.group_by("inv_secondary").agg(pl.len().alias("n")).sort("n", descending=True)
       .collect())
print("inv_secondary value counts (top 30):")
print(sec.head(30))

wc = (lf.group_by("weapon_code").agg(pl.len().alias("n")).sort("n", descending=True)
      .head(20).collect())
print("weapon_code value counts (top 20):")
print(wc)

rd = pl.scan_parquet(ROUNDS)
print("ROUNDS schema:")
for k, v in rd.collect_schema().items():
    print("  ", k, v)
print("rounds rows:", rd.select(pl.len()).collect().item())
print(rd.head(3).collect())

bt = pl.scan_parquet(BUYS)
print("BUYS schema:")
for k, v in bt.collect_schema().items():
    print("  ", k, v)
print("buys rows:", bt.select(pl.len()).collect().item())
print(bt.head(3).collect())
