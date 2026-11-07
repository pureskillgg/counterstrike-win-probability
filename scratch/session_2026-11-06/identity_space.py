"""Run ON the space (~/venv-dsdk/bin/python identity_space.py). Compares recomputed bucket chunks with the existing ones.

Env: POSB_OUT = folder with the recomputed bucket_cNNN.parquet files (the existing ones are in final/pos_parts).
Exact comparison of every feature column (NaN == NaN), and identical (match_key, round, tick) keys.
"""
import os
import sys

import numpy as np
import polars as pl

FIN = os.path.expanduser("~/adam/builder/final")
NEW = os.environ["POSB_OUT"]
KEYS = ["match_key", "round", "tick"]
ok_all = True
for c in sys.argv[1:]:
    a = pl.read_parquet(f"{FIN}/pos_parts/bucket_{c}.parquet").sort(KEYS)
    b = pl.read_parquet(f"{NEW}/bucket_{c}.parquet").sort(KEYS)
    same_shape = a.height == b.height and a.columns == b.columns
    same_keys = same_shape and bool((a.select(KEYS).to_numpy() == b.select(KEYS).to_numpy()).all())
    diffs = {}
    if same_keys:
        for col in a.columns:
            if col in KEYS:
                continue
            x = a[col].cast(pl.Float64).to_numpy()
            y = b[col].cast(pl.Float64).to_numpy()
            diffs[col] = int((~((np.isnan(x) & np.isnan(y)) | (x == y))).sum())
    bad = {k: v for k, v in diffs.items() if v}
    ok = same_shape and same_keys and not bad
    ok_all &= ok
    print(f"{c}: rows {a.height:,} vs {b.height:,} | shape {same_shape} keys {same_keys} | differing values: {bad if bad else 'none'} | {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
print("ALL IDENTICAL" if ok_all else "NOT IDENTICAL", flush=True)
