import polars as pl, numpy as np, sys
print("polars", pl.__version__, "numpy", np.__version__)
base = "G:/pureskill/csds-cs2/data"
files = {
 "per_tick_states": f"{base}/viz/per_tick_states.parquet",
 "per_tick_test_preds": f"{base}/viz/per_tick_test_preds.parquet",
 "players_1hz": f"{base}/state_sample/players_1hz.parquet",
 "rounds": f"{base}/state_sample/rounds.parquet",
}
for k, f in files.items():
    lf = pl.scan_parquet(f)
    print("=====", k)
    print(lf.collect_schema())
    print("rows:", lf.select(pl.len()).collect().item())
