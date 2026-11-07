"""verify_E1_4 bench: independent recomputation of the E1-4 CPU thread sweep.

Same workload as scripts\\swarm2\\E1_threads.py (100,000 x 50 float32, seed 7, 10 trees, depth 4,
eta 0.1, hist, binary:logistic). Workload parameters are ASSUMPTIONS carried over from E1 (not re-derived).

Differences from the claimed E1 run (on purpose):
  * more timed repetitions (5 per setting, not 2), so the spread is visible
  * configs are interleaved in a shuffled order each round, so host drift is not confounded with nthread
  * each timed run records: wall time, process CPU time (all threads), CPU/wall ratio,
    system busy % (GetSystemTimes sampled every 0.2 s), and process page-fault delta
  * 'child' mode runs one nthread setting in a fresh process (no cross-setting state)

Modes:
  sweep : one process, shuffled interleaved rounds over nthread in [1,2,4,8,16]
  child : one process, one nthread setting, prints a single JSON line (prefix 'JSON ')

Run (project interpreter):
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E1_4_bench.py sweep --rounds 3
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E1_4_bench.py child --nthread 8 --reps 5
"""
import argparse
import ctypes
import json
import os
import random
import sys
import threading
import time
from ctypes import wintypes

import numpy as np
import xgboost as xgb

N = 100_000  # ASSUMPTION (from E1_threads.py)
F = 50  # ASSUMPTION
TREES = 10  # ASSUMPTION
WARM = 2  # ASSUMPTION: untimed warm-up trees per setting
NTHREADS = [1, 2, 4, 8, 16]
BASE = {"objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
        "max_depth": 4, "eta": 0.1, "seed": 0}

# ---------- Windows host counters via ctypes (no psutil in these envs) ----------
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.GetCurrentProcess.argtypes = []
kernel32.GetSystemTimes.restype = wintypes.BOOL
kernel32.GetSystemTimes.argtypes = [ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
                                    ctypes.POINTER(wintypes.FILETIME)]


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
psapi.EnumProcessModules.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE),
                                     wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
psapi.EnumProcessModules.restype = wintypes.BOOL
psapi.GetModuleFileNameExW.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
psapi.GetModuleFileNameExW.restype = wintypes.DWORD


def _ft(f):
    return (f.dwHighDateTime << 32) | f.dwLowDateTime


def sys_times():
    idle, kern, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    # GetSystemTimes: kernel time INCLUDES idle time.
    return _ft(idle), _ft(kern) + _ft(user)


def busy_pct(t0, t1):
    idle = t1[0] - t0[0]
    total = t1[1] - t0[1]
    return 100.0 * (total - idle) / total if total > 0 else float("nan")


def pf_count():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(pmc)
    psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PageFaultCount


def loaded_modules():
    hproc = kernel32.GetCurrentProcess()
    arr = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(hproc, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
        return []
    n = min(needed.value // ctypes.sizeof(wintypes.HMODULE), len(arr))
    buf = ctypes.create_unicode_buffer(520)
    out = []
    for i in range(n):
        psapi.GetModuleFileNameExW(hproc, arr[i], buf, 520)
        out.append(buf.value)
    return out


class HostSampler(threading.Thread):
    """Samples system busy % every `interval` seconds in the background."""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []  # (perf_counter, busy_pct)
        self._stop_evt = threading.Event()
        self._prev = sys_times()

    def run(self):
        while not self._stop_evt.wait(self.interval):
            cur = sys_times()
            self.samples.append((time.perf_counter(), busy_pct(self._prev, cur)))
            self._prev = cur

    def stop(self):
        self._stop_evt.set()
        self.join()

    def mean_between(self, t0, t1):
        vals = [b for (t, b) in self.samples if t0 <= t <= t1 and b == b]
        return sum(vals) / len(vals) if vals else float("nan")


def make_data():
    rng = np.random.default_rng(7)  # same seed and draw order as E1_threads.py
    X = rng.standard_normal((N, F), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(N).astype(np.float32) > 0).astype(np.float32)
    return X, y


def timed_train(params, dmat, sampler, rep_index, tag):
    pf0 = pf_count()
    st0 = sys_times()
    c0 = time.process_time()
    t0 = time.perf_counter()
    xgb.train(params, dmat, num_boost_round=TREES)
    t1 = time.perf_counter()
    c1 = time.process_time()
    st1 = sys_times()
    pf1 = pf_count()
    wall = t1 - t0
    cpu = c1 - c0
    run = {
        "tag": tag,
        "nthread": params["nthread"],
        "rep": rep_index,
        "wall_s": round(wall, 4),
        "s_per_tree": round(wall / TREES, 5),
        "proc_cpu_s": round(cpu, 4),
        "cpu_over_wall": round(cpu / wall, 3) if wall > 0 else None,
        # direct: system busy % over exactly this run window (GetSystemTimes tick ~15.6 ms)
        "host_busy_window_pct": round(busy_pct(st0, st1), 2),
        # sampled: mean of 0.2 s samples inside the window (NaN if the window is shorter than a sample)
        "host_busy_sampler_pct": round(sampler.mean_between(t0, t1), 2),
        "page_faults": pf1 - pf0,
        "t_start_perf": round(t0, 3),
        "t_end_perf": round(t1, 3),
    }
    return run


def run_sweep(rounds, tag):
    X, y = make_data()
    d = xgb.DMatrix(X, label=y)
    sampler = HostSampler()
    sampler.start()
    runs = []
    print(f"SWEEP tag={tag} python={sys.executable} xgboost={xgb.__version__} "
          f"USE_CUDA={xgb.build_info().get('USE_CUDA')} rounds={rounds} trees={TREES} rows={N} feats={F}",
          flush=True)
    for r in range(rounds):
        order = NTHREADS[:]
        random.Random(1000 + r).shuffle(order)  # interleaved, reproducible order per round
        for nt in order:
            params = dict(BASE, nthread=nt)
            xgb.train(params, d, num_boost_round=WARM)  # untimed warm-up
            run = timed_train(params, d, sampler, r, tag)
            runs.append(run)
            print(f"RUN round={r} nthread={nt:<3} wall={run['wall_s']:9.3f}s "
                  f"s/tree={run['s_per_tree']:8.4f} cpu/wall={run['cpu_over_wall']} "
                  f"busy%={run['host_busy_window_pct']} pf={run['page_faults']}", flush=True)
    sampler.stop()
    summary = {"mode": "sweep", "tag": tag, "python": sys.executable, "xgboost": xgb.__version__,
               "use_cuda": xgb.build_info().get("USE_CUDA"), "rounds": rounds,
               "omp_modules": [m for m in loaded_modules() if any(k in m.lower() for k in
                                                                    ("omp", "gomp", "iomp", "mkl"))],
               "env_omp": {k: v for k, v in os.environ.items() if k.upper().startswith(("OMP_", "KMP_"))},
               "runs": runs}
    print("JSON " + json.dumps(summary), flush=True)


def run_child(nthread, reps, tag):
    X, y = make_data()
    d = xgb.DMatrix(X, label=y)
    sampler = HostSampler()
    sampler.start()
    params = dict(BASE, nthread=nthread)
    xgb.train(params, d, num_boost_round=WARM)  # untimed warm-up
    runs = [timed_train(params, d, sampler, i, tag) for i in range(reps)]
    sampler.stop()
    summary = {"mode": "child", "tag": tag, "nthread": nthread, "python": sys.executable,
               "xgboost": xgb.__version__, "use_cuda": xgb.build_info().get("USE_CUDA"),
               "omp_modules": [m for m in loaded_modules() if any(k in m.lower() for k in
                                                                    ("omp", "gomp", "iomp", "mkl"))],
               "env_omp": {k: v for k, v in os.environ.items() if k.upper().startswith(("OMP_", "KMP_"))},
               "runs": runs}
    print("JSON " + json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sp = sub.add_parser("sweep")
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--tag", default="sweep")
    cp = sub.add_parser("child")
    cp.add_argument("--nthread", type=int, required=True)
    cp.add_argument("--reps", type=int, default=5)
    cp.add_argument("--tag", default="child")
    args = ap.parse_args()
    if args.mode == "sweep":
        run_sweep(args.rounds, args.tag)
    else:
        run_child(args.nthread, args.reps, args.tag)


if __name__ == "__main__":
    main()
