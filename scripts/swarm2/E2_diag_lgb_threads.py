"""E2 diagnostic: LightGBM CPU fit time versus num_threads, no GPU sampler running.

Run with the cs2gpu_alt interpreter only:
  python -I -X utf8 E2_diag_lgb_threads.py
Prints one line per thread count. Writes nothing to disk.
"""
import sys
import time

import numpy as np
import lightgbm as lgb

rng = np.random.default_rng(1)
n, p = 20000, 20
X = rng.standard_normal((n, p), dtype=np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n) > 0).astype(np.int32)

print("python", sys.version.split()[0], "lightgbm", lgb.__version__, flush=True)
for threads in (1, 2, 4):
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                  seed=0, verbose=-1, device_type="cpu", num_threads=threads)
    t0 = time.perf_counter()
    ds = lgb.Dataset(X, label=y, params=params).construct()
    t1 = time.perf_counter()
    lgb.train(params, ds, num_boost_round=20)
    t2 = time.perf_counter()
    print(f"num_threads={threads} build={t1 - t0:.3f}s train_20_trees={t2 - t1:.3f}s", flush=True)
