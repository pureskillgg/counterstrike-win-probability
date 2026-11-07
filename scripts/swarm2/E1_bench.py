"""E1: synthetic XGBoost benchmark, CPU vs device=cuda, in the new C: cs2gpu env.

Workload (all ASSUMPTION, not taken from the project):
  rows = 1,000,000 train + 100,000 holdout; features = 50 float32 N(0,1);
  binary:logistic target with a nonlinear logit plus noise;
  tree_method = hist, max_depth = 4, num_boost_round = 200, eta = 0.1.
Configurations:
  cpu_default : device=cpu, xgboost default threads (16 logical CPUs). 1 timed run (too slow to repeat).
  cpu_1thread : device=cpu, nthread=1. 2 timed runs.
  cuda        : device=cuda (default threads). 3 timed runs.
Each configuration: one untimed warm-up (5 rounds), then the timed runs; the median is reported.
The machine was at ~100% CPU from other processes during this run (see gpu/cpu snapshots).

Run:
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\E1_bench.py
"""
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import xgboost as xgb

N_TRAIN = 1_000_000  # ASSUMPTION: benchmark size from the task
N_HOLD = 100_000  # ASSUMPTION
N_FEAT = 50  # ASSUMPTION
N_TREES = 200  # from task
MAX_DEPTH = 4  # from task
ETA = 0.1  # ASSUMPTION
WARM_ROUNDS = 5  # ASSUMPTION
SEED = 12345  # ASSUMPTION

CONFIGS = [
    # name, device, nthread, reps
    ("cpu_default", "cpu", None, 1),
    ("cpu_1thread", "cpu", 1, 2),
    ("cuda", "cuda", None, 3),
]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_psapi = ctypes.WinDLL("psapi")
_k32 = ctypes.WinDLL("kernel32")
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
_psapi.GetProcessMemoryInfo.restype = ctypes.c_int


def mem_mb():
    """Return (working set, peak working set) of this process in MiB."""
    c = PROCESS_MEMORY_COUNTERS()
    c.cb = ctypes.sizeof(c)
    if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        return (-1.0, -1.0)
    return (round(c.WorkingSetSize / 2**20, 1), round(c.PeakWorkingSetSize / 2**20, 1))


def snapshot(tag):
    """nvidia-smi memory/util and host CPU load, for contention context."""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        gpu = f"nvidia-smi failed: {e}"
    try:
        import subprocess as sp
        cpu = sp.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        cpu = f"cpu probe failed: {e}"
    line = f"[{tag}] gpu(mem,util)={gpu} | host_cpu_load%={cpu} | ws,peak_MB={mem_mb()}"
    print(line, flush=True)
    return line


def make_data():
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_HOLD
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
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def params_for(device, nthread):
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": device,
        "max_depth": MAX_DEPTH,
        "eta": ETA,
        "seed": 0,
    }
    if nthread is not None:
        p["nthread"] = nthread
    return p


def logloss(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    yy = np.asarray(y, dtype=np.float64)
    return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))


def bench_config(name, device, nthread, reps, dtrain, dhold, yhold):
    res = {"config": name, "device": device, "nthread": nthread, "reps": reps}
    t0 = time.perf_counter()
    xgb.train(params_for(device, nthread), dtrain, num_boost_round=WARM_ROUNDS)
    res["warmup_5rounds_s"] = round(time.perf_counter() - t0, 3)
    snapshot(f"{name} before timed runs")

    times, booster = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        booster = xgb.train(params_for(device, nthread), dtrain, num_boost_round=N_TREES)
        times.append(time.perf_counter() - t0)
        print(f"[{name}] run {len(times)}/{reps}: {times[-1]:.3f} s", flush=True)
    snapshot(f"{name} after timed runs")

    res["train_times_s"] = [round(t, 3) for t in times]
    res["train_median_s"] = round(statistics.median(times), 3)
    res["train_min_s"] = round(min(times), 3)
    res["s_per_tree_median"] = round(statistics.median(times) / N_TREES, 4)

    cfg = json.loads(booster.save_config())
    res["config_device"] = cfg["learner"]["generic_param"].get("device")
    res["config_nthread"] = cfg["learner"]["generic_param"].get("nthread")
    res["num_boosted_rounds"] = booster.num_boosted_rounds()

    t0 = time.perf_counter()
    p_hold = booster.predict(dhold)
    res["predict_holdout_s"] = round(time.perf_counter() - t0, 3)
    res["holdout_logloss"] = round(logloss(p_hold, yhold), 6)
    res["peak_rss_mb_so_far"] = mem_mb()[1]
    return res, p_hold


def main():
    print("python", sys.version.split()[0], sys.executable, flush=True)
    print("platform", platform.platform(), flush=True)
    print("xgboost", xgb.__version__, flush=True)
    bi = xgb.build_info()
    print("USE_CUDA", bi.get("USE_CUDA"), "CUDA_VERSION", bi.get("CUDA_VERSION"),
          "USE_OPENMP", bi.get("USE_OPENMP"), flush=True)
    print("logical_cpus", os.cpu_count(), flush=True)
    snapshot("start")

    t0 = time.perf_counter()
    Xtr, ytr, Xho, yho = make_data()
    print(f"data generated in {time.perf_counter() - t0:.2f} s; X_train {Xtr.shape} {Xtr.dtype} "
          f"{Xtr.nbytes / 2**20:.1f} MiB; ws,peak_MB={mem_mb()}", flush=True)

    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dhold = xgb.DMatrix(Xho)
    print(f"DMatrix build {time.perf_counter() - t0:.3f} s; ws,peak_MB={mem_mb()}", flush=True)

    results, preds = {}, {}
    for name, device, nthread, reps in CONFIGS:
        r, p = bench_config(name, device, nthread, reps, dtrain, dhold, yho)
        results[name] = r
        preds[name] = p

    base = results["cpu_1thread"]["train_median_s"]
    diffs = {}
    ref = preds["cuda"].astype(np.float64)
    for name in ("cpu_default", "cpu_1thread"):
        d = np.abs(preds[name].astype(np.float64) - ref)
        diffs[f"{name}_vs_cuda"] = {"max_abs": float(d.max()), "mean_abs": float(d.mean())}

    summary = {
        "versions": {"python": sys.version.split()[0], "xgboost": xgb.__version__, "numpy": np.__version__},
        "xgboost_build": {"USE_CUDA": bi.get("USE_CUDA"), "CUDA_VERSION": str(bi.get("CUDA_VERSION"))},
        "workload": {
            "n_train": N_TRAIN, "n_holdout": N_HOLD, "n_features": N_FEAT, "n_trees": N_TREES,
            "max_depth": MAX_DEPTH, "eta": ETA, "tree_method": "hist", "warmup_rounds": WARM_ROUNDS,
            "seed": SEED,
        },
        "results": results,
        "speedup_cuda_vs_cpu_default_median": round(
            results["cpu_default"]["train_median_s"] / results["cuda"]["train_median_s"], 3),
        "speedup_cuda_vs_cpu_1thread_median": round(base / results["cuda"]["train_median_s"], 3),
        "holdout_pred_diff": diffs,
        "peak_rss_mb_end": mem_mb()[1],
    }
    snapshot("end")
    print("SUMMARY_JSON_BEGIN", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print("SUMMARY_JSON_END", flush=True)


if __name__ == "__main__":
    main()
