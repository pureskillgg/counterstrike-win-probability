"""verify_E2_5 bench: one CPU configuration, one fit, one process per call (same convention as E2_bench.py).

Usage (E2 workspace interpreter; the project env has no catboost):
  <cs2gpu_alt>\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E2_5_bench.py <variant> <rep>

Data and claim settings replicate scripts\\swarm2\\E2_bench.py: seed 20241, 100000 train, 50000 valid,
50 float32 features, 100 rounds/iterations, 1 CPU thread. Variants change one thing each (see VARIANTS).
Writes nothing to disk. Prints one line: VERIFY_E2_5_JSON {...}
"""
import ctypes
import json
import sys
import threading
import time
import traceback
from ctypes import wintypes

import numpy as np

SEED = 20241
N_TRAIN, N_VALID, N_FEAT, ITERS = 100000, 50000, 50, 100

# ASSUMPTION: "claim_*" = the E2 settings exactly. Other variants change one thing so the
# effect of each default can be read off separately.
VARIANTS = {
    "claim_lgb": ("lgb", {"num_leaves": 63, "max_bin": 255}),
    "lgb_l64_b127": ("lgb", {"num_leaves": 64, "max_bin": 127}),
    "claim_cb": ("cb", {}),
    "cb_iter1": ("cb", {"iterations": 1}),
    "cb_le1": ("cb", {"leaf_estimation_iterations": 1}),
    "cb_le1_nobag": ("cb", {"leaf_estimation_iterations": 1, "bootstrap_type": "No"}),
}

variant = sys.argv[1]
rep = int(sys.argv[2])
engine, extra = VARIANTS[variant]


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def _ft(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def system_times():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    # kernel time includes idle time on Windows
    return _ft(idle), _ft(kern) + _ft(user)


def peak_rss_mb():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    pmc = PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 2 ** 20, 1)


class CpuSampler:
    """Whole-system busy % every 0.25 s (GetSystemTimes). Includes every other job on the machine."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._last = system_times()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            idle, total = system_times()
            d_idle = idle - self._last[0]
            d_total = total - self._last[1]
            self._last = (idle, total)
            if d_total > 0:
                self.samples.append((time.perf_counter(), 100.0 * (1.0 - d_idle / d_total)))
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def busy_in(self, t0, t1):
        vals = [b for ts, b in self.samples if t0 <= ts <= t1]
        return round(sum(vals) / len(vals), 1) if vals else None


def timed(fn):
    w0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    w1, c1 = time.perf_counter(), time.process_time()
    return out, w0, w1, round(w1 - w0, 4), round(c1 - c0, 4)


def make_data():
    # Identical to scripts\swarm2\E2_bench.py make_data().
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_VALID
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    f = (1.0 * X[:, 0] - 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.sin(2.0 * X[:, 3])
         + 0.5 * (X[:, 4] > 0) * X[:, 5] - 0.3 * X[:, 6] ** 2 + 0.2 * X[:, 7]).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-f))
    y = (rng.random(n) < p).astype(np.int32)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def logloss(y_true, p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-15, 1 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def main():
    rec = {"variant": variant, "rep": rep, "engine": engine, "extra": extra, "status": "error"}
    try:
        if engine == "lgb":
            import lightgbm as lgb
            rec["lib_version"] = lgb.__version__
        else:
            import catboost as cb
            rec["lib_version"] = cb.__version__
        Xtr, ytr, Xva, yva = make_data()
        rec["train_positive_rate"] = round(float(ytr.mean()), 4)
        sampler = CpuSampler()
        sampler.start()
        time.sleep(1.0)
        if engine == "lgb":
            params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255, seed=0,
                          verbose=-1, device_type="cpu", num_threads=1)
            params.update(extra)
            rec["params"] = params
            dtr, b0, b1, build_wall, build_cpu = timed(
                lambda: lgb.Dataset(Xtr, label=ytr, params=params, free_raw_data=False).construct())
            booster, f0, f1, fit_wall, fit_cpu = timed(
                lambda: lgb.train(params, dtr, num_boost_round=ITERS))
            pred, _, _, pred_wall, _ = timed(lambda: booster.predict(Xva))
            rec.update(build_wall=build_wall, build_cpu=build_cpu, fit_wall=fit_wall, fit_cpu=fit_cpu,
                       predict_wall=pred_wall, trees=int(booster.num_trees()),
                       busy_fit_pct=sampler.busy_in(f0, f1), busy_build_pct=sampler.busy_in(b0, b1))
        else:
            params = dict(iterations=ITERS, depth=6, learning_rate=0.1, loss_function="Logloss",
                          border_count=128, random_seed=0, verbose=0, allow_writing_files=False,
                          task_type="CPU", thread_count=1)
            params.update(extra)
            rec["params"] = params
            model = cb.CatBoostClassifier(**params)
            _, f0, f1, fit_wall, fit_cpu = timed(lambda: model.fit(Xtr, ytr))
            pred, _, _, pred_wall, _ = timed(lambda: model.predict_proba(Xva)[:, 1])
            eff = model.get_all_params()
            rec.update(fit_wall=fit_wall, fit_cpu=fit_cpu, predict_wall=pred_wall,
                       trees=int(model.tree_count_), busy_fit_pct=sampler.busy_in(f0, f1),
                       eff_boosting_type=str(eff.get("boosting_type")),
                       eff_bootstrap_type=str(eff.get("bootstrap_type")),
                       eff_leaf_estimation_iterations=eff.get("leaf_estimation_iterations"),
                       eff_leaf_estimation_method=str(eff.get("leaf_estimation_method")))
        rec["logloss"] = round(logloss(yva, pred), 6)
        rec["fit_cpu_over_wall"] = round(rec["fit_cpu"] / rec["fit_wall"], 3) if rec["fit_wall"] else None
        sampler.stop()
        rec["sampler_samples"] = len(sampler.samples)
        rec["peak_rss_mb"] = peak_rss_mb()
        rec["status"] = "ok"
    except Exception as exc:  # record the failure instead of hiding it
        rec["error_repr"] = repr(exc)
        traceback.print_exc()
    print("VERIFY_E2_5_JSON " + json.dumps(rec, default=str), flush=True)


if __name__ == "__main__":
    main()
