"""verify_E1_2 (isolation test): can the CPU/CUDA model gap be removed by making the quantile cuts identical?

Idea: if features have very few distinct values, both builds may keep every distinct value as a cut. When the cut
sets are identical (checked, value by value), any remaining CPU/CUDA difference must come from something other than
the sketch (for example histogram accumulation or split tie-breaking). If the cut sets are not identical, the test is
inconclusive for that step size.

Data: same synthetic generator and labels as verify_E1_2_synth.py (seed 12345, 100k train, 100k holdout), features
rounded to a step (1.0 or 0.5). Params as E1: hist, depth 4, eta 0.1, 200 trees, no subsampling.
Interpreter (needs CUDA): C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_2_isolate.py
"""
import json
import sys

import numpy as np
import xgboost as xgb

SEED, N_TRAIN, N_HOLD, N_FEAT = 12345, 100_000, 100_000, 50


def make(step):
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_HOLD
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    logit = (0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + 0.4 * np.sin(2.0 * X[:, 4])
             + 0.3 * (X[:, 5] > 0) + 0.5 * rng.standard_normal(n).astype(np.float32)).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    Xq = (np.round(X / step) * step).astype(np.float32)
    return Xq[:N_TRAIN], y[:N_TRAIN], Xq[N_TRAIN:], y[N_TRAIN:]


def params(device, nthread=None):
    p = {"objective": "binary:logistic", "tree_method": "hist", "device": device, "max_depth": 4, "eta": 0.1, "seed": 0}
    if nthread is not None:
        p["nthread"] = nthread
    return p


def cuts(d):
    ptr, val = d.get_quantile_cut()
    return np.asarray(ptr), np.asarray(val, dtype=np.float64)


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def run(step):
    Xtr, ytr, Xho, yho = make(step)
    d_cpu = xgb.DMatrix(Xtr, label=ytr)
    b_cpu = xgb.train(params("cpu", nthread=1), d_cpu, num_boost_round=200)
    pc = b_cpu.predict(xgb.DMatrix(Xho))
    cpu_ptr, cpu_val = cuts(d_cpu)
    d_gpu = xgb.DMatrix(Xtr, label=ytr)
    b_gpu = xgb.train(params("cuda"), d_gpu, num_boost_round=200)
    pg = b_gpu.predict(xgb.DMatrix(Xho))
    gpu_ptr, gpu_val = cuts(d_gpu)

    same_ptr = cpu_ptr.shape == gpu_ptr.shape and bool(np.array_equal(cpu_ptr, gpu_ptr))
    same_val_shape = cpu_val.shape == gpu_val.shape
    cut_identical = bool(same_ptr and same_val_shape and np.array_equal(cpu_val, gpu_val))
    if same_val_shape:
        fin = np.isfinite(cpu_val) & np.isfinite(gpu_val)
    else:
        fin = np.zeros(0, dtype=bool)
    distinct = [len(np.unique(Xtr[:, j])) for j in range(N_FEAT)]
    cut_counts_cpu = np.diff(cpu_ptr) - 1  # minus the +inf sentinel per feature
    cut_counts_gpu = np.diff(gpu_ptr) - 1
    d = np.abs(pc.astype(np.float64) - pg.astype(np.float64))
    rep = {"step": step, "distinct_values_per_feature_min_max": [min(distinct), max(distinct)],
           "cut_counts_cpu_total": int(cut_counts_cpu.sum()), "cut_counts_cuda_total": int(cut_counts_gpu.sum()),
           "features_with_same_cut_count": int(np.sum(cut_counts_cpu == cut_counts_gpu)),
           "cut_arrays_identical": cut_identical,
           "finite_cut_max_abs_diff": (float(np.max(np.abs(cpu_val[fin] - gpu_val[fin])))
                                       if same_val_shape and fin.any() else "n/a (cut array shapes differ)"),
           "logloss_cpu": logloss(pc, yho), "logloss_cuda": logloss(pg, yho),
           "pred_max_abs_diff": float(d.max()), "pred_mean_abs_diff": float(d.mean()),
           "pred_rows_differing": int(np.count_nonzero(d))}
    print("[isolate]", json.dumps(rep), flush=True)


if __name__ == "__main__":
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    steps = [float(s) for s in sys.argv[1:]] or [1.0, 0.5]
    for s in steps:
        run(s)
