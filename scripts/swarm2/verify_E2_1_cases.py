"""Verifier for E2-1 (swarm2): one LightGBM case per process, read-only, writes nothing to disk.

A native OpenCL or CUDA abort kills the interpreter, so each case runs in its own process.
The caller records the exit code from the shell and redirects output to logs\\swarm2\\verify_E2_1.log.

Cases:
  ctl_abort       control: os.abort() (exit code mapping check)
  ctl_av          control: read address 0 (access violation; exit code mapping check)
  cpu_tiny        device_type=cpu, 2000 rows, 5 trees, num_threads=4 (same data as E2_env_check)
  cuda_tiny       device_type=cuda, same data (claim: CUDA refused at build level)
  gpu_default     device_type=gpu, LightGBM default gpu_platform_id/gpu_device_id (-1, -1)
  gpu_plat0       device_type=gpu, gpu_platform_id=0, gpu_device_id=0
  gpu_plat1       device_type=gpu, gpu_platform_id=1, gpu_device_id=0
  cpu_bench       device_type=cpu, the E2_bench CPU configuration (100k train, 50k valid, 50 feat,
                  100 trees, num_leaves 63, max_bin 255, 1 thread, 2 reps); recomputes the claim's
                  median fit and validation logloss on the same synthetic data
Usage (cs2gpu_alt interpreter only):
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe -I -X utf8 verify_E2_1_cases.py <case>
"""
import ctypes
import faulthandler
import json
import os
import statistics
import sys
import time
import traceback

faulthandler.enable(all_threads=True)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402

case = sys.argv[1] if len(sys.argv) > 1 else "cpu_tiny"
print(f"VERIFY_E2_1_CASE_START {case}", flush=True)
print("python:", sys.executable, flush=True)

if case == "ctl_abort":
    os.abort()
if case == "ctl_av":
    ctypes.string_at(0)  # read at address 0 -> access violation
    raise SystemExit("ctl_av did not crash")

import lightgbm as lgb  # noqa: E402

print("lightgbm", lgb.__version__, flush=True)


def logloss(y_true, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


result = {"case": case, "lightgbm": lgb.__version__}
try:
    if case in ("cpu_tiny", "cuda_tiny", "gpu_default", "gpu_plat0", "gpu_plat1"):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2000, 10)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(2000) > 0).astype(np.int32)
        params = {"objective": "binary", "verbose": 2, "num_leaves": 15, "seed": 0}
        if case == "cpu_tiny":
            params.update(device_type="cpu", num_threads=4, verbose=-1)
        elif case == "cuda_tiny":
            params.update(device_type="cuda")
        elif case == "gpu_default":
            params.update(device_type="gpu")
        elif case == "gpu_plat0":
            params.update(device_type="gpu", gpu_platform_id=0, gpu_device_id=0)
        elif case == "gpu_plat1":
            params.update(device_type="gpu", gpu_platform_id=1, gpu_device_id=0)
        t0 = time.perf_counter()
        booster = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=5)
        result.update(status="ok", num_trees=int(booster.num_trees()),
                      train_seconds=round(time.perf_counter() - t0, 3),
                      predict_head=[round(float(v), 6) for v in booster.predict(X[:5])])
    elif case in ("cpu_bench", "gpu_plat1_bench"):
        n_train, n_valid, n_feat, iters, repeats = 100000, 50000, 50, 100, 2
        if case == "gpu_plat1_bench":
            repeats = 1  # scope check only: does the NVIDIA OpenCL path train at benchmark size
        rng = np.random.default_rng(20241)
        n = n_train + n_valid
        X = rng.standard_normal((n, n_feat), dtype=np.float32)
        f = (1.0 * X[:, 0] - 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.sin(2.0 * X[:, 3])
             + 0.5 * (X[:, 4] > 0) * X[:, 5] - 0.3 * X[:, 6] ** 2 + 0.2 * X[:, 7]).astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-f))
        y = (rng.random(n) < p).astype(np.int32)
        Xtr, ytr, Xva, yva = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
        if case == "gpu_plat1_bench":
            params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                          seed=0, verbose=2, device_type="gpu", gpu_platform_id=1, gpu_device_id=0)
        else:
            params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                          seed=0, verbose=-1, device_type="cpu", num_threads=1)
        reps = []
        for rep in range(repeats):
            dtr = lgb.Dataset(Xtr, label=ytr, params=params, free_raw_data=False).construct()
            t0 = time.perf_counter()
            booster = lgb.train(params, dtr, num_boost_round=iters)
            t1 = time.perf_counter()
            pred = booster.predict(Xva)
            reps.append({"rep": rep, "fit_seconds": round(t1 - t0, 3),
                         "valid_logloss": round(logloss(yva, pred), 6),
                         "n_trees": int(booster.num_trees())})
            print("rep", reps[-1], flush=True)
        result.update(status="ok", reps=reps,
                      median_fit_seconds=round(statistics.median(r["fit_seconds"] for r in reps), 4),
                      train_positive_rate=round(float(ytr.mean()), 4))
    else:
        raise SystemExit(f"unknown case {case}")
except SystemExit:
    raise
except Exception as exc:  # LightGBMError and friends
    result["status"] = "error"
    result["error_repr"] = repr(exc)
    traceback.print_exc()

print("VERIFY_E2_1_JSON " + json.dumps(result, default=str), flush=True)
print(f"VERIFY_E2_1_CASE_END {case}", flush=True)
