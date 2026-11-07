"""VERIFY E2-3, independent pass: one CatBoost device per process, claim-size data.

Usage (cs2gpu_alt interpreter, catboost 1.2.10, the one the E2 runs used):
  "C:\\Users\\13144\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe" -I -X utf8 verify_E2_3_indep_bench.py <CPU|GPU> <iters_csv>

Data: 100000 train rows x 50 features, same generator and seed as scripts\\swarm2\\E2_bench.py.
Settings: depth 6, learning_rate 0.1, border_count 128, Logloss, random_seed 0.
  CPU: thread_count=1 (the claim's CPU arm).
  GPU: task_type GPU, devices 0, gpu_ram_part 0.2, one untimed 5-tree warm-up on 20k rows (as in E2).
Timing: perf_counter around model.fit only. System CPU busy % comes from GetSystemTimes over
exactly the same window. GPU memory and utilisation come from nvidia-smi sampled every 0.5 s during the window.
Prints one 'INDEP_FIT {json}' line per fit. Writes nothing to disk.
"""
import ctypes
import json
import statistics
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import numpy as np
import catboost as cb

DEVICE = sys.argv[1].upper()
ITERS = [int(v) for v in sys.argv[2].split(",")]
SEED = 20241
N_TRAIN, N_VALID, N_FEAT = 100000, 50000, 50
GPU_RAM_PART = 0.2  # ASSUMPTION: same as E2_bench.py
CPU_THREADS = 1     # ASSUMPTION: same as the claim's CPU arm


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.GetSystemTimes.restype = wintypes.BOOL
_k32.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME)]


def _ft(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def sys_times():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    if not _k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
        raise OSError(ctypes.get_last_error(), "GetSystemTimes failed")
    # kernel time includes idle time on Windows
    return _ft(idle), _ft(kern) + _ft(user)


def busy_between(a, b):
    d_total = b[1] - a[1]
    d_idle = b[0] - a[0]
    return round(100.0 * (1.0 - d_idle / d_total), 1) if d_total > 0 else None


def gpu_query():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    mem, util = (float(v) for v in out.split(","))
    return mem, util


class GpuSampler:
    def __init__(self, interval=0.5):
        self.samples = []  # (perf_counter time, memory.used MiB, utilization.gpu %)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                mem, util = gpu_query()
                self.samples.append((time.perf_counter(), mem, util))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=30)

    def window(self, t0, t1, pre=3.0):
        base = [s[1] for s in self.samples if t0 - pre <= s[0] < t0]
        inside = [s for s in self.samples if t0 <= s[0] <= t1]
        if not inside:
            return {"gpu_samples": 0, "gpu_mem_pre_min_mib": min(base) if base else None,
                    "gpu_mem_max_mib": None, "gpu_util_mean_pct": None, "gpu_util_max_pct": None}
        return {"gpu_samples": len(inside),
                "gpu_mem_pre_min_mib": min(base) if base else None,
                "gpu_mem_max_mib": max(s[1] for s in inside),
                "gpu_util_mean_pct": round(statistics.mean(s[2] for s in inside), 1),
                "gpu_util_max_pct": max(s[2] for s in inside)}


def make_data():
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_VALID
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    f = (1.0 * X[:, 0] - 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.sin(2.0 * X[:, 3])
         + 0.5 * (X[:, 4] > 0) * X[:, 5] - 0.3 * X[:, 6] ** 2 + 0.2 * X[:, 7]).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-f))
    y = (rng.random(n) < p).astype(np.int32)
    return X[:N_TRAIN], y[:N_TRAIN]


def params_for(iters):
    p = dict(iterations=iters, depth=6, learning_rate=0.1, loss_function="Logloss",
             border_count=128, random_seed=0, verbose=0, allow_writing_files=False)
    if DEVICE == "GPU":
        p.update(task_type="GPU", devices="0", gpu_ram_part=GPU_RAM_PART)
    else:
        p.update(task_type="CPU", thread_count=CPU_THREADS)
    return p


def main():
    X, y = make_data()
    meta = {"device": DEVICE, "iters": ITERS, "catboost": cb.__version__, "n_train": int(len(y)),
            "n_feat": int(X.shape[1]), "train_positive_rate": round(float(y.mean()), 4),
            "gpu_ram_part": GPU_RAM_PART if DEVICE == "GPU" else None,
            "cpu_thread_count": CPU_THREADS if DEVICE == "CPU" else None,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("INDEP_META " + json.dumps(meta), flush=True)

    if DEVICE == "GPU":
        w = cb.CatBoostClassifier(**params_for(5))
        t0 = time.perf_counter()
        w.fit(X[:20000], y[:20000])
        print(f"INDEP_WARMUP {json.dumps({'device': 'GPU', 'warmup_seconds': round(time.perf_counter() - t0, 3)})}",
              flush=True)

    sampler = GpuSampler()
    sampler.start()
    time.sleep(1.0)
    try:
        for it in ITERS:
            model = cb.CatBoostClassifier(**params_for(it))
            rec = {"device": DEVICE, "iters": it, "error": None}
            a = sys_times()
            t0 = time.perf_counter()
            try:
                model.fit(X, y)
            except Exception as exc:  # record and continue
                rec["error"] = repr(exc)[:300]
            t1 = time.perf_counter()
            b = sys_times()
            rec.update(fit_seconds=round(t1 - t0, 3), system_cpu_busy_pct=busy_between(a, b),
                       n_trees=int(model.tree_count_) if rec["error"] is None else None,
                       task_type_effective=str(model.get_all_params().get("task_type")) if rec["error"] is None else None)
            rec.update(sampler.window(t0, t1))
            print("INDEP_FIT " + json.dumps(rec), flush=True)
    finally:
        sampler.stop()
    print("INDEP_DONE " + time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
