import polars as pl
import sys
print("polars", pl.__version__)
g = r"G:\pureskill\csds-cs2\data\viz\per_tick_states.parquet"
p = r"G:\pureskill\csds-cs2\data\viz\per_tick_test_preds.parquet"
for f in [g, p]:
    lf = pl.scan_parquet(f)
    print("FILE", f)
    print(lf.collect_schema())
    print("rows", lf.select(pl.len()).collect().item())
    print(lf.head(5).collect())
