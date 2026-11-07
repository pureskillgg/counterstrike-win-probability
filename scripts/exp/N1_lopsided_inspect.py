import sys
import polars as pl
import numpy as np
print("python", sys.version.split()[0])
print("polars", pl.__version__)
print("numpy", np.__version__)
base = r"G:\pureskill\csds-cs2\data"
files = {
    "per_tick_states": base + r"\viz\per_tick_states.parquet",
    "per_tick_test_preds": base + r"\viz\per_tick_test_preds.parquet",
    "players_1hz": base + r"\state_sample\players_1hz.parquet",
    "rounds": base + r"\state_sample\rounds.parquet",
}
for name, path in files.items():
    lf = pl.scan_parquet(path)
    schema = lf.collect_schema()
    n = lf.select(pl.len()).collect().item()
    print("=== ", name, "rows=", n)
    for c, t in schema.items():
        print("   ", c, t)
    print(lf.head(3).collect())
