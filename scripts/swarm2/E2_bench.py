"""E2 synthetic benchmark: CatBoost and LightGBM, CPU and GPU, one configuration per process.

Usage (cs2gpu_alt interpreter only):
  python -I -X utf8 E2_bench.py <engine cb|lgb> <device> <n_train> <n_valid> <n_feat> <iters> <repeats> [gpu_ram_part]
  device: CPU/GPU for cb; cpu/gpu/cuda for lgb.

Same synthetic data (seed 20241, float32 features, binary target from a fixed nonlinear logit) for every engine.
Hyperparameters are identical across engines where a counterpart exists (ASSUMPTION, chosen by E2):
  depth 6 (cb) ~ num_leaves 63 (lgb), learning_rate 0.1, border_count 128 (cb) / max_bin 255 (lgb),
  Logloss / binary, random_seed 0, 16 CPU threads for CPU runs, GPU device 0 for GPU runs.
Writes nothing to disk; the caller redirects stdout/stderr to logs\\swarm2\\E2_bench_*.log.
"""
import ctypes
import json
import statistics
import subprocess
import sys
import threading
import time
import traceback
from ctypes import wintypes

import numpy as np

engine = sys.argv[1]
device = sys.argv[2]
n_train, n_valid, n_feat, iters, repeats = (int(v) for v in sys.argv[3:8])
gpu_ram_part = float(sys.argv[8]) if len(sys.argv) > 8 else 0.2
SEED = 20241
# ASSUMPTION: 1 CPU thread for every CPU baseline. The machine ran at ~100% CPU from other swarm jobs.
# E2_diag_lgb_threads.py (20k rows, 20 trees): 1 thread 0.586 s, 2 threads 2.039 s, 4 threads 17.004 s.
# With 16 threads the smoke test took 70.263 s for 20 trees on 20k rows. Extra OpenMP threads hurt here.
CPU_THREADS = 1


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.GetCurrentProcess.argtypes = []
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]


def peak_rss_mb():
    """Peak working set of this process in MiB; returns None if the Win32 call fails."""
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    ok = _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    if not ok:
        return None
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


def _ft_to_int(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def system_cpu_times():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    # kernel time includes idle time on Windows
    return _ft_to_int(idle), _ft_to_int(kern) + _ft_to_int(user)


class GpuSampler:
    """Polls nvidia-smi (memory.used, utilization.gpu) and whole-system CPU busy % (GetSystemTimes).
    WDDM hides per-process GPU memory, so the GPU view includes the other project jobs on the same card.
    The CPU view includes every other swarm job on the machine."""

    def __init__(self, interval=0.5):
        self.samples = []
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_cpu = system_cpu_times()

    def _query(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
            mem, util = (float(v) for v in out.split(","))
            return mem, util
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            q = self._query()
            idle, total = system_cpu_times()
            d_idle = idle - self._last_cpu[0]
            d_total = total - self._last_cpu[1]
            self._last_cpu = (idle, total)
            busy = 100.0 * (1.0 - d_idle / d_total) if d_total > 0 else None
            if q is not None:
                self.samples.append((time.perf_counter(), q[0], q[1], busy))
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=20)

    def window(self, t_start, t_end, pre_seconds=4.0):
        base = [s[1] for s in self.samples if t_start - pre_seconds <= s[0] < t_start]
        inside = [s for s in self.samples if t_start <= s[0] <= t_end]
        cpu = [s[3] for s in inside if s[3] is not None]
        if not inside:
            return {"gpu_mem_baseline_mib": None, "gpu_mem_peak_mib": None,
                    "gpu_util_mean_pct": None, "gpu_util_max_pct": None, "gpu_samples": 0,
                    "system_cpu_busy_mean_pct": None}
        return {
            "gpu_mem_baseline_mib": min(base) if base else None,
            "gpu_mem_peak_mib": max(s[1] for s in inside),
            "gpu_util_mean_pct": round(statistics.mean(s[2] for s in inside), 1),
            "gpu_util_max_pct": max(s[2] for s in inside),
            "gpu_samples": len(inside),
            "system_cpu_busy_mean_pct": round(statistics.mean(cpu), 1) if cpu else None,
        }


def make_data():
    rng = np.random.default_rng(SEED)
    n = n_train + n_valid
    X = rng.standard_normal((n, n_feat), dtype=np.float32)
    f = (1.0 * X[:, 0] - 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.sin(2.0 * X[:, 3])
         + 0.5 * (X[:, 4] > 0) * X[:, 5] - 0.3 * X[:, 6] ** 2 + 0.2 * X[:, 7]).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-f))
    y = (rng.random(n) < p).astype(np.int32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def logloss(y_true, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def run_catboost(Xtr, ytr, Xva, yva, rec, sampler):
    import catboost as cb
    params = dict(iterations=iters, depth=6, learning_rate=0.1, loss_function="Logloss",
                  border_count=128, random_seed=0, verbose=0, allow_writing_files=False)
    if device.upper() == "GPU":
        params.update(task_type="GPU", devices="0", gpu_ram_part=gpu_ram_part)
    else:
        params.update(task_type="CPU", thread_count=CPU_THREADS)
    rec["params"] = {k: v for k, v in params.items()}

    if device.upper() == "GPU":
        print("stage: catboost GPU warm-up start", flush=True)
        warm = cb.CatBoostClassifier(**{**params, "iterations": 5})
        w0 = time.perf_counter()
        warm.fit(Xtr[:20000], ytr[:20000])
        rec["gpu_warmup_seconds"] = round(time.perf_counter() - w0, 3)
        print(f"stage: catboost GPU warm-up done in {rec['gpu_warmup_seconds']} s", flush=True)

    for rep in range(repeats):
        r = {"rep": rep}
        print(f"stage: catboost rep {rep} fit start", flush=True)
        t0 = time.perf_counter()
        model = cb.CatBoostClassifier(**params)
        model.fit(Xtr, ytr)
        t1 = time.perf_counter()
        pred = model.predict_proba(Xva)[:, 1]
        t2 = time.perf_counter()
        r.update(fit_seconds=round(t1 - t0, 3), predict_seconds=round(t2 - t1, 3),
                 valid_logloss=round(logloss(yva, pred), 6), n_trees=int(model.tree_count_),
                 task_type_effective=str(model.get_all_params().get("task_type")))
        r.update(sampler.window(t0, t1))
        r["peak_rss_mb"] = peak_rss_mb()
        rec.setdefault("reps", []).append(r)
        print(f"rep {rep}: fit {r['fit_seconds']} s, predict {r['predict_seconds']} s, "
              f"logloss {r['valid_logloss']}, trees {r['n_trees']}", flush=True)


def run_lightgbm(Xtr, ytr, Xva, yva, rec, sampler):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                  seed=0, verbose=-1, device_type=device.lower())
    if device.lower() == "cpu":
        params["num_threads"] = CPU_THREADS
    rec["params"] = params

    for rep in range(repeats):
        r = {"rep": rep}
        b0 = time.perf_counter()
        dtr = lgb.Dataset(Xtr, label=ytr, params=params, free_raw_data=False).construct()
        b1 = time.perf_counter()
        t0 = time.perf_counter()
        booster = lgb.train(params, dtr, num_boost_round=iters)
        t1 = time.perf_counter()
        pred = booster.predict(Xva)
        t2 = time.perf_counter()
        r.update(build_seconds=round(b1 - b0, 3), fit_seconds=round(t1 - t0, 3),
                 predict_seconds=round(t2 - t1, 3), valid_logloss=round(logloss(yva, pred), 6),
                 n_trees=int(booster.num_trees()))
        r.update(sampler.window(t0, t1))
        r["peak_rss_mb"] = peak_rss_mb()
        rec.setdefault("reps", []).append(r)
        print(f"rep {rep}: build {r['build_seconds']} s, fit {r['fit_seconds']} s, "
              f"predict {r['predict_seconds']} s, logloss {r['valid_logloss']}, trees {r['n_trees']}",
              flush=True)


def main():
    rec = {"engine": engine, "device": device, "n_train": n_train, "n_valid": n_valid,
           "n_feat": n_feat, "iters": iters, "repeats": repeats, "seed": SEED,
           "gpu_ram_part": gpu_ram_part if device.upper() == "GPU" and engine == "cb" else None}
    import importlib
    lib = importlib.import_module("catboost" if engine == "cb" else "lightgbm")
    rec["library_version"] = lib.__version__
    Xtr, ytr, Xva, yva = make_data()
    rec["train_positive_rate"] = round(float(ytr.mean()), 4)
    rec["valid_positive_rate"] = round(float(yva.mean()), 4)
    sampler = GpuSampler()
    sampler.start()
    time.sleep(2.0)
    try:
        if engine == "cb":
            run_catboost(Xtr, ytr, Xva, yva, rec, sampler)
        else:
            run_lightgbm(Xtr, ytr, Xva, yva, rec, sampler)
        rec["status"] = "ok"
    except Exception as exc:
        rec["status"] = "error"
        rec["error_repr"] = repr(exc)
        traceback.print_exc()
    finally:
        sampler.stop()
    if rec.get("reps"):
        for key in ("fit_seconds", "predict_seconds", "valid_logloss"):
            vals = [r[key] for r in rec["reps"]]
            rec[f"median_{key}"] = round(statistics.median(vals), 6)
    print("E2_BENCH_JSON " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
