"""How much faster is the GPU than the CPU for the same boosted-tree fit?   (CUDA XGBoost environment)

  ...tabicl\\python.exe -I -X utf8 scripts\\model\\bench_gpu_cpu.py [--rows 1500000] [--trees 300]

Identical settings on real training rows (the 59 inputs of the final model, depth 4, learning rate 0.05, histogram method, a fixed number of trees, no early stopping), once with
device='cuda' and once with device='cpu' on all available threads; 2 repeats each, best time kept. Writes data/<store>/bench_gpu_cpu.json. The timings are of THIS machine, and other jobs
may be running on it: the file records the thread count and the load note.
"""
import os

import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=1500000)
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--store", default="combined")
a = ap.parse_args()
feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
d = os.path.join(W.ROOT, "data", a.store, "cache_snapshot_2026.parquet")
G = pl.scan_parquet(d).select(feats + ["label_ct_win"]).filter(pl.col("seconds_since_round_live") >= 0).head(a.rows).collect()
X = G.select(feats).to_numpy().astype(np.float32)
y = G["label_ct_win"].to_numpy().astype(np.int64)
print(f"{len(y):,} rows x {X.shape[1]} inputs, {a.trees} trees", flush=True)
res = {"rows": int(len(y)), "inputs": int(X.shape[1]), "trees": a.trees, "cpu_threads": os.cpu_count(), "runs": {}}
for dev in ("cuda", "cpu"):
    best = 1e9
    for _ in range(2):
        m = xgb.XGBClassifier(n_estimators=a.trees, learning_rate=0.05, max_depth=4, tree_method="hist", device=dev, subsample=0.8, min_child_weight=20, random_state=0, n_jobs=os.cpu_count())
        t0 = time.time()
        m.fit(X, y)
        best = min(best, time.time() - t0)
    p = m.predict_proba(X[:200000])[:, 1]
    ll = float(-np.mean(y[:200000] * np.log(np.clip(p, 1e-6, 1)) + (1 - y[:200000]) * np.log(np.clip(1 - p, 1e-6, 1))))
    res["runs"][dev] = {"seconds": round(best, 1), "train_logloss_first_200k": round(ll, 4)}
    print(dev, res["runs"][dev], flush=True)
res["speedup"] = round(res["runs"]["cpu"]["seconds"] / res["runs"]["cuda"]["seconds"], 1)
res["note"] = "measured on one laptop (RTX 3080 20 GB, this CPU) while other jobs were running; a quiet machine can only be faster for both"
W.dump_json(res, os.path.join(W.ROOT, "data", a.store, "bench_gpu_cpu.json"))
print("speed-up", res["speedup"])
