import polars as pl
g = pl.scan_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_states.parquet")
p = pl.scan_parquet(r"G:\pureskill\csds-cs2\data\viz\per_tick_test_preds.parquet")
print("GRID SCHEMA:")
for k, v in g.collect_schema().items():
    print("  ", k, v)
print("PRED SCHEMA:")
for k, v in p.collect_schema().items():
    print("  ", k, v)
gd = g.head(5).collect()
print(gd)
pd_ = p.head(5).collect()
print(pd_)
print("grid rows:", g.select(pl.len()).collect().item())
print("pred rows:", p.select(pl.len()).collect().item())
