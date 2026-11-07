"""E1: CPU vs CUDA holdout parity check on the same synthetic data as E1_bench.py.

Trains 200 trees, depth 4, hist, eta 0.1 with device=cpu (nthread=1, to avoid
OpenMP oversubscription on the busy host) and device=cuda, at 100k and 1M train rows.
Reports holdout log loss and prediction differences. Workload is an ASSUMPTION
(same synthetic generator as E1_bench.py, same seed, 50 float32 features).

Run:
  C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\E1_parity.py
"""
import time

import numpy as np
import xgboost as xgb

SEED = 12345  # ASSUMPTION (same as E1_bench.py)
N_FEAT = 50  # ASSUMPTION
N_HOLD = 100_000  # ASSUMPTION
N_TREES = 200  # from task
MAX_DEPTH = 4  # from task
ETA = 0.1  # ASSUMPTION


def make_data(n_train):
    rng = np.random.default_rng(SEED)
    n = n_train + N_HOLD
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    logit = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.5 * X[:, 2] * X[:, 3]
        + 0.4 * np.sin(2.0 * X[:, 4])
        + 0.3 * (X[:, 5] > 0)
        + 0.5 * rng.standard_normal(n).astype(np.float32)
    ).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def run(n_train):
    Xtr, ytr, Xho, yho = make_data(n_train)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dho = xgb.DMatrix(Xho)
    out = {}
    for name, dev, nth in (("cpu_1thread", "cpu", 1), ("cuda", "cuda", None)):
        params = {"objective": "binary:logistic", "tree_method": "hist", "device": dev,
                  "max_depth": MAX_DEPTH, "eta": ETA, "seed": 0}
        if nth is not None:
            params["nthread"] = nth
        t0 = time.perf_counter()
        b = xgb.train(params, dtr, num_boost_round=N_TREES)
        dt = time.perf_counter() - t0
        p = b.predict(dho)
        out[name] = (p, dt)
        print(f"n_train={n_train:>9,} {name:<12} time={dt:8.3f}s holdout_logloss={logloss(p, yho):.6f}",
              flush=True)
    pc, pg = out["cpu_1thread"][0].astype(np.float64), out["cuda"][0].astype(np.float64)
    d = np.abs(pc - pg)
    print(f"n_train={n_train:>9,} cpu_vs_cuda max_abs={d.max():.6f} mean_abs={d.mean():.6f} "
          f"frac_abs_gt_0.01={np.mean(d > 0.01):.4f} logloss_gap(cuda-cpu)="
          f"{logloss(pg, yho) - logloss(pc, yho):+.6f}", flush=True)


if __name__ == "__main__":
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    run(100_000)
    run(1_000_000)
