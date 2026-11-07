"""verify_E1_4: independent recheck of "CPU threading collapses under host load".

Same workload as scripts\\swarm2\\E1_threads.py (ASSUMPTIONS carried over):
100,000 rows x 50 float32, binary:logistic, tree_method hist, max_depth 4, eta 0.1,
seed 0, 2 warm-up trees (not timed), 10 timed trees.

Differences from E1_threads.py (to separate the claimed cause from confounds):
  * thread configs are interleaved in a random order per block (ASSUMPTION: seeded
    shuffle), so host-load drift is not aligned with the thread count;
  * 4 blocks per config instead of 2 reps, and the xgboost default (no nthread) is included;
  * per timed run: process CPU seconds (kernel+user), system-wide busy fraction
    (GetSystemTimes), and a single-core probe (fixed pure-Python loop) to see how fast
    one core is running at that moment;
  * the OpenMP runtime DLLs the process actually loaded are listed.

Run (project interpreter, explicit):
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 ^
      scripts\\swarm2\\verify_E1_4_sweep.py --blocks 4 --log logs\\swarm2\\verify_E1_4_sweep.log
  Optional load-test flags: --rows 1000000 --configs 1,4,default
  (logs\\swarm2\\verify_E1_4.log belongs to another writer and is not used by this script.)
"""
import argparse
import ctypes
import ctypes.wintypes as wt
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import xgboost as xgb

N = 100_000  # ASSUMPTION (as E1_threads.py)
F = 50  # ASSUMPTION
TREES = 10  # ASSUMPTION
WARM = 2  # ASSUMPTION
PROBE_ITERS = 2_000_000  # ASSUMPTION: single-core probe size
CONFIGS = [1, 2, 4, 8, 16, None]  # None = xgboost default (no nthread param)


class FILETIME(ctypes.Structure):
    _fields_ = [("lo", wt.DWORD), ("hi", wt.DWORD)]


def _ft(f):
    return (f.hi << 32) | f.lo  # 100-ns units


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.GetSystemTimes.argtypes = [ctypes.POINTER(FILETIME)] * 3
_k32.GetSystemTimes.restype = wt.BOOL
_k32.GetProcessTimes.argtypes = [wt.HANDLE] + [ctypes.POINTER(FILETIME)] * 4
_k32.GetProcessTimes.restype = wt.BOOL
_psapi.EnumProcessModules.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_void_p), wt.DWORD, ctypes.POINTER(wt.DWORD)]
_psapi.EnumProcessModules.restype = wt.BOOL
_psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.LPWSTR, wt.DWORD]
_psapi.GetModuleFileNameExW.restype = wt.DWORD

LOG_FH = None


def out(msg=""):
    print(msg, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(msg + "\n")
        LOG_FH.flush()


def sys_times():
    i, k, u = FILETIME(), FILETIME(), FILETIME()
    if not _k32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u)):
        raise OSError(ctypes.get_last_error(), "GetSystemTimes failed")
    idle = _ft(i)
    total = _ft(k) + _ft(u)  # kernel time already includes idle time
    return idle, total


def proc_cpu_seconds():
    c, e, k, u = FILETIME(), FILETIME(), FILETIME(), FILETIME()
    h = _k32.GetCurrentProcess()
    if not _k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
        raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
    return (_ft(k) + _ft(u)) / 1e7


def loaded_omp_modules():
    arr = (ctypes.c_void_p * 4096)()
    needed = wt.DWORD()
    h = _k32.GetCurrentProcess()
    if not _psapi.EnumProcessModules(h, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
        return ["<EnumProcessModules failed>"]
    n = min(needed.value // ctypes.sizeof(ctypes.c_void_p), 4096)
    buf = ctypes.create_unicode_buffer(1024)
    names = []
    for i in range(n):
        if _psapi.GetModuleFileNameExW(h, arr[i], buf, 1024):
            p = buf.value
            low = p.lower()
            if any(tag in low for tag in ("omp", "gomp", "iomp", "xgboost", "vcomp")):
                names.append(p)
    return names


def probe():
    t0 = time.perf_counter()
    x = 0
    for i in range(PROBE_ITERS):
        x += i * i % 7
    return time.perf_counter() - t0


def make_data():
    rng = np.random.default_rng(7)  # same as E1_threads.py
    X = rng.standard_normal((N, F), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(N).astype(np.float32) > 0).astype(np.float32)
    return X, y


def timed_run(d, nthread, ncpu):
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
              "max_depth": 4, "eta": 0.1, "seed": 0}
    if nthread is not None:
        params["nthread"] = nthread
    xgb.train(params, d, num_boost_round=WARM)  # warm-up, not timed
    idle0, tot0 = sys_times()
    c0 = proc_cpu_seconds()
    w0 = time.perf_counter()
    xgb.train(params, d, num_boost_round=TREES)
    wall = time.perf_counter() - w0
    c1 = proc_cpu_seconds()
    idle1, tot1 = sys_times()
    own = c1 - c0
    dtot = (tot1 - tot0) / 1e7
    busy = 1.0 - ((idle1 - idle0) / (tot1 - tot0)) if tot1 > tot0 else float("nan")
    return {
        "wall": wall,
        "s_per_tree": wall / TREES,
        "own_cpu": own,
        "eff_par": own / wall if wall > 0 else float("nan"),
        "own_frac": own / (wall * ncpu) if wall > 0 else float("nan"),
        "sys_busy": busy,
        "sys_window_s": dtot,
    }


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def lab(nt):
    return "default" if nt is None else str(nt)


def parse_configs(s):
    out_list = []
    for tok in s.split(","):
        tok = tok.strip()
        out_list.append(None if tok == "default" else int(tok))
    return out_list


def main():
    global LOG_FH, N, CONFIGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--rows", type=int, default=100_000)  # ASSUMPTION default matches E1_threads.py
    ap.add_argument("--configs", default="1,2,4,8,16,default")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    N = args.rows
    CONFIGS = parse_configs(args.configs)
    if args.log:
        os.makedirs(os.path.dirname(args.log), exist_ok=True)
        LOG_FH = open(args.log, "w", encoding="utf-8", newline="\n")

    ncpu = os.cpu_count() or 1
    out("== verify_E1_4_sweep")
    out(f"python {sys.executable}")
    out(f"python {sys.version.split()[0]}  xgboost {xgb.__version__}  USE_CUDA={xgb.build_info().get('USE_CUDA')}"
        f"  USE_OPENMP={xgb.build_info().get('USE_OPENMP')}")
    out(f"os.cpu_count()={ncpu}  OMP env: " + ", ".join(
        f"{k}={os.environ.get(k)}" for k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "OMP_DYNAMIC", "OMP_PROC_BIND",
                                                "GOMP_SPINCOUNT", "KMP_BLOCKTIME")))
    out(f"workload: N={N} F={F} TREES={TREES} WARM={WARM} blocks={args.blocks} configs={[lab(c) for c in CONFIGS]}")
    t_start = time.time()
    X, y = make_data()
    d = xgb.DMatrix(X, label=y)
    xgb.train({"objective": "binary:logistic", "tree_method": "hist", "device": "cpu", "max_depth": 4,
               "eta": 0.1, "seed": 0}, d, num_boost_round=1)  # load the runtime before listing modules
    out("loaded OpenMP / xgboost modules: " + " | ".join(loaded_omp_modules()))
    out(f"idle probe (no load of ours): {probe():.4f} s for {PROBE_ITERS} iters")
    out("")

    rows = []
    for b in range(args.blocks):
        order = CONFIGS[:]
        random.Random(1000 + b).shuffle(order)  # ASSUMPTION: seeded random order per block
        out(f"-- block {b} order: {[lab(c) for c in order]}")
        for nt in order:
            r = timed_run(d, nt, ncpu)
            r["probe_s"] = probe()
            r["block"] = b
            r["nthread"] = nt
            rows.append(r)
            out(f"block={b} nthread={lab(nt):<8} wall={r['wall']:8.3f}s s_per_tree={r['s_per_tree']:8.4f} "
                f"own_cpu={r['own_cpu']:8.2f}s eff_par={r['eff_par']:6.2f} own_frac={r['own_frac']:6.3f} "
                f"sys_busy={r['sys_busy']:6.3f} probe={r['probe_s']:.4f}s")
    out("")

    out("== per-config summary (s_per_tree over the timed 10 trees)")
    med = {}
    for nt in CONFIGS:
        rs = [r for r in rows if r["nthread"] == nt]
        spt = [r["s_per_tree"] for r in rs]
        med[nt] = statistics.median(spt)
        out(f"nthread={lab(nt):<8} n={len(rs)} median={med[nt]:.4f} min={min(spt):.4f} max={max(spt):.4f} "
            f"max/min={max(spt) / min(spt):6.2f} eff_par_median={statistics.median([r['eff_par'] for r in rs]):6.2f} "
            f"sys_busy_median={statistics.median([r['sys_busy'] for r in rs]):6.3f} "
            f"probe_median={statistics.median([r['probe_s'] for r in rs]):.4f}s")
    out("")
    out("== ratios (median s_per_tree relative to nthread=1)")
    for nt in CONFIGS:
        out(f"nthread={lab(nt):<8} ratio={med[nt] / med[1]:8.2f}")
    out(f"claimed ratios from E1_threads.log: nthread=2 -> {1.4304 / 0.0502:.2f}, nthread=16 -> {1.1606 / 0.0502:.2f}")
    out("")

    out("== association checks (log s_per_tree vs covariates, all runs unless stated)")
    ls = [math.log(r["s_per_tree"]) for r in rows]
    out(f"corr(log s/tree, sys_busy)           all={pearson(ls, [r['sys_busy'] for r in rows]):+.3f}")
    out(f"corr(log s/tree, probe_s)            all={pearson(ls, [r['probe_s'] for r in rows]):+.3f}")
    multi = [r for r in rows if r["nthread"] != 1]
    lm = [math.log(r["s_per_tree"]) for r in multi]
    out(f"corr(log s/tree, sys_busy)           multi-thread only (n={len(multi)})={pearson(lm, [r['sys_busy'] for r in multi]):+.3f}")
    out(f"corr(log s/tree, probe_s)            multi-thread only={pearson(lm, [r['probe_s'] for r in multi]):+.3f}")
    one = [r for r in rows if r["nthread"] == 1]
    out(f"nthread=1 runs: s/tree={[round(r['s_per_tree'], 4) for r in one]} sys_busy={[round(r['sys_busy'], 3) for r in one]}")
    out("")
    out(f"elapsed {time.time() - t_start:.1f} s")
    if LOG_FH is not None:
        LOG_FH.close()


if __name__ == "__main__":
    main()
