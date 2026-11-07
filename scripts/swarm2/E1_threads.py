"""E1: CPU thread-count sweep under the current system load (short runs).

Workload is an ASSUMPTION: 100,000 rows x 50 float32 features, 10 trees, depth 4,
hist, eta 0.1. One warm-up run per setting, then the median of 2 timed runs.
Also times one CUDA run of the same workload for reference.

Run:
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\E1_threads.py
"""
import statistics
import time

import numpy as np
import xgboost as xgb

N = 100_000  # ASSUMPTION
F = 50  # ASSUMPTION
TREES = 10  # ASSUMPTION
REPS = 2  # ASSUMPTION


def main():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((N, F), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(N).astype(np.float32) > 0).astype(np.float32)
    d = xgb.DMatrix(X, label=y)
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)

    settings = [("cpu", 1), ("cpu", 2), ("cpu", 4), ("cpu", 8), ("cpu", 16), ("cuda", None)]
    for device, nthread in settings:
        params = {"objective": "binary:logistic", "tree_method": "hist", "device": device,
                  "max_depth": 4, "eta": 0.1, "seed": 0}
        if nthread is not None:
            params["nthread"] = nthread
        xgb.train(params, d, num_boost_round=2)  # warm-up
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            xgb.train(params, d, num_boost_round=TREES)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        print(f"device={device:<4} nthread={str(nthread):<5} median_s={med:8.3f} "
              f"s_per_tree={med / TREES:8.4f} runs={[round(t, 3) for t in times]}", flush=True)


if __name__ == "__main__":
    main()
