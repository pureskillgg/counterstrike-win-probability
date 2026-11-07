"""Child process for the E2-1 independent probe: one tiny LightGBM training run.

Usage (cs2gpu_alt interpreter only):
  python -I -X utf8 verify_E2_1_indep_child.py <device> <threads> <n_rows> <iters> [gpu_platform_id gpu_device_id]

Prints one line "PROBE_JSON {...}" only if the Python call returns or raises a
Python exception. A native crash, a hang, or a kill produces no PROBE_JSON line.
ASSUMPTION: synthetic data (20 features, binary label), seed 20241, num_leaves 63,
max_bin 255, learning_rate 0.1, same as the E2 bench parameters except size.
"""
import json
import sys
import time

import numpy as np
import lightgbm as lgb


def main():
    dev = sys.argv[1]
    threads = int(sys.argv[2])
    n = int(sys.argv[3])
    iters = int(sys.argv[4])
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                  seed=0, verbose=-1, device_type=dev)
    if dev == "cpu":
        params["num_threads"] = threads
    if dev == "gpu" and len(sys.argv) > 6:
        params["gpu_platform_id"] = int(sys.argv[5])
        params["gpu_device_id"] = int(sys.argv[6])
    rng = np.random.default_rng(20241)
    X = rng.standard_normal((n, 20))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.5 * rng.standard_normal(n) > 0).astype(np.int32)
    rec = {"device": dev, "threads": threads if dev == "cpu" else None, "n_rows": n, "iters": iters,
           "lightgbm": lgb.__version__, "gpu_platform_id": params.get("gpu_platform_id"),
           "gpu_device_id": params.get("gpu_device_id")}
    t0 = time.perf_counter()
    try:
        ds = lgb.Dataset(X, label=y, params=params, free_raw_data=False).construct()
        t1 = time.perf_counter()
        booster = lgb.train(params, ds, num_boost_round=iters)
        t2 = time.perf_counter()
        pred = booster.predict(X[:1000])
        rec.update(status="ok", build_s=round(t1 - t0, 3), fit_s=round(t2 - t1, 3),
                   trees=booster.num_trees(), pred_mean=round(float(pred.mean()), 6))
    except Exception as exc:
        rec.update(status="error", error=repr(exc)[:300])
    print("PROBE_JSON " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
