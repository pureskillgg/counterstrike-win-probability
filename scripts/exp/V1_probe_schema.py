import polars as pl
import os

BASE = r"G:\pureskill\csds-cs2\data"
files = [
    os.path.join(BASE, "state_sample", "players_1hz.parquet"),
    os.path.join(BASE, "state_sample", "rounds.parquet"),
    os.path.join(BASE, "viz", "per_tick_states.parquet"),
    os.path.join(BASE, "viz", "per_tick_test_preds.parquet"),
]
for f in files:
    print("=== ", f, "exists=", os.path.exists(f))
    if not os.path.exists(f):
        continue
    lf = pl.scan_parquet(f)
    schema = lf.collect_schema()
    for k, v in schema.items():
        print("   ", k, v)
    df = lf.head(3).collect()
    print("  head rows:")
    with pl.Config(tbl_cols=40, tbl_width_chars=250):
        print(df)
    print("  row count:", lf.select(pl.len()).collect().item())
