"""E1 diagnostic: where does CPU hist training start failing with 'bad allocation'?

Scales the row count on CPU with a short run (20 trees, depth 4) and reports
the peak working set of this process after each step. Also times one CUDA run
at the same sizes for comparison. Workload numbers are ASSUMPTIONS.

Run:
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\E1_diag.py
"""
import ctypes
import sys
import time

import numpy as np
import xgboost as xgb

N_FEAT = 50  # ASSUMPTION
TREES = 20  # ASSUMPTION


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
    c = PROCESS_MEMORY_COUNTERS()
    c.cb = ctypes.sizeof(c)
    ok = _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    if not ok:
        return (-1.0, -1.0)
    return (round(c.WorkingSetSize / 2**20, 1), round(c.PeakWorkingSetSize / 2**20, 1))


def run(n, device, nthread=None):
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n).astype(np.float32) > 0).astype(np.float32)
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": device, "max_depth": 4, "eta": 0.1, "seed": 0}
    if nthread is not None:
        params["nthread"] = nthread
    d = xgb.DMatrix(X, label=y)
    before = mem_mb()
    t0 = time.perf_counter()
    try:
        xgb.train(params, d, num_boost_round=TREES)
        status = "OK"
    except Exception as e:  # noqa: BLE001
        status = f"FAILED {type(e).__name__}: {str(e)[:160]}"
    dt = time.perf_counter() - t0
    after = mem_mb()
    print(f"n={n:>9,} device={device:<4} nthread={str(nthread):<4} {status:<60} "
          f"time={dt:7.2f}s  ws_before={before[0]}MB ws_after={after[0]}MB peak={after[1]}MB", flush=True)
    del X, y, d
    return status


if __name__ == "__main__":
    print("python", sys.version.split()[0], "xgboost", xgb.__version__, flush=True)
    print("start mem (ws, peak) MB =", mem_mb(), flush=True)
    for n in (100_000, 300_000, 1_000_000):
        run(n, "cpu")
    print("--- CPU with nthread=4 at 1M ---", flush=True)
    run(1_000_000, "cpu", nthread=4)
    print("--- CUDA at 1M ---", flush=True)
    run(1_000_000, "cuda")
