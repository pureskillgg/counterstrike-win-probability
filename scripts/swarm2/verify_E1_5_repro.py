"""verify E1-5 reproduction, run under CURRENT host conditions (no induced memory pressure).

Two checks, both with the project interpreter (G: env, xgboost 3.4.2 CPU build):
  1. 1M-row CPU hist training (same generator and seed as E1_parity.py / E1_bench.py,
     20 trees instead of 200 to keep the run short: ASSUMPTION), in isolated child processes,
     with progress markers so the phase of any native crash is visible. Each child's peak
     working set and peak commit (PeakPagefileUsage) are recorded. The parent records
     the child's exit code (hex) and the system commit before and after.
  2. Repeated import-startup probes (numpy, xgboost, sklearn, polars, pyarrow) in fresh
     processes, to test whether the startup-phase 0xC000070A fault recurs now.

Not run: any induced commit exhaustion (would risk other processes on this shared host).

Run:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_E1_5_repro.py
"""
import ctypes
import subprocess
import sys
import time
from ctypes import wintypes

PY = r"G:\pureskill\csds-cs2\.pixi\envs\default\python.exe"
SELF = r"G:\pureskill\csds-cs2\scripts\swarm2\verify_E1_5_repro.py"
SEED = 12345  # ASSUMPTION (same as E1_parity.py)
N_TRAIN = 1_000_000  # from the claimed 1M step
N_HOLD = 100_000  # ASSUMPTION
N_FEAT = 50  # ASSUMPTION
N_TREES = 20  # ASSUMPTION (shortened from 200)
MAX_DEPTH = 4
ETA = 0.1
GB = 2**30


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", wintypes.DWORD),
                ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


psapi = ctypes.WinDLL("psapi")
k32 = ctypes.WinDLL("kernel32")
psapi.GetPerformanceInfo.argtypes = [ctypes.POINTER(PERFORMANCE_INFORMATION), wintypes.DWORD]
psapi.GetPerformanceInfo.restype = wintypes.BOOL
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
k32.GetCurrentProcess.restype = wintypes.HANDLE


def system_commit():
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    ps = pi.PageSize
    return pi.CommitTotal * ps / GB, pi.CommitLimit * ps / GB, pi.PhysicalAvailable * ps / GB


def self_mem():
    c = PROCESS_MEMORY_COUNTERS_EX()
    c.cb = ctypes.sizeof(c)
    psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 2**20, c.PeakWorkingSetSize / 2**20, c.PrivateUsage / 2**20, c.PeakPagefileUsage / 2**20


def make_data():
    import numpy as np
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_HOLD
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    logit = (0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + 0.4 * np.sin(2.0 * X[:, 4])
             + 0.3 * (X[:, 5] > 0) + 0.5 * rng.standard_normal(n).astype(np.float32)).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def child(nthread):
    t0 = time.perf_counter()
    import numpy as np
    import xgboost as xgb
    print(f"[child] imports done in {time.perf_counter() - t0:.2f} s; xgboost {xgb.__version__} USE_CUDA={xgb.build_info().get('USE_CUDA')}", flush=True)
    print(f"[child] mem ws/peakws/private/peakcommit MB = {tuple(round(x, 1) for x in self_mem())}", flush=True)
    t0 = time.perf_counter()
    Xtr, ytr, Xho, yho = make_data()
    print(f"[child] data generated {time.perf_counter() - t0:.2f} s, X_train {Xtr.shape}", flush=True)
    t0 = time.perf_counter()
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dho = xgb.DMatrix(Xho)
    print(f"[child] DMatrix built {time.perf_counter() - t0:.2f} s; mem={tuple(round(x, 1) for x in self_mem())}", flush=True)
    params = {"objective": "binary:logistic", "tree_method": "hist", "device": "cpu", "max_depth": MAX_DEPTH,
              "eta": ETA, "seed": 0, "nthread": nthread}
    print(f"[child] train start nthread={nthread} rounds={N_TREES}", flush=True)
    t0 = time.perf_counter()
    try:
        b = xgb.train(params, dtr, num_boost_round=N_TREES)
    except Exception as e:  # noqa: BLE001
        print(f"[child] EXCEPTION during train: {type(e).__name__}: {str(e)[:300]}", flush=True)
        print(f"[child] mem at exception ws/peakws/private/peakcommit MB = {tuple(round(x, 1) for x in self_mem())}", flush=True)
        return 3
    dt = time.perf_counter() - t0
    print(f"[child] train done {dt:.2f} s", flush=True)
    p = b.predict(dho).astype(np.float64)
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    ll = float(-np.mean(yho * np.log(pc) + (1 - yho) * np.log(1 - pc)))
    print(f"[child] holdout logloss {ll:.6f}; mem ws/peakws/private/peakcommit MB = {tuple(round(x, 1) for x in self_mem())}", flush=True)
    return 0


def run_child(nthread):
    before = system_commit()
    t0 = time.perf_counter()
    r = subprocess.run([PY, "-I", "-X", "utf8", SELF, "--child", str(nthread)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    wall = time.perf_counter() - t0
    after = system_commit()
    print(f"\n== child nthread={nthread}: exit={r.returncode} (hex 0x{r.returncode & 0xFFFFFFFF:08X}) wall={wall:.1f} s")
    print(f"   commit before: {before[0]:.2f}/{before[1]:.2f} GB, free RAM {before[2]:.2f} GB")
    print(f"   commit after : {after[0]:.2f}/{after[1]:.2f} GB, free RAM {after[2]:.2f} GB")
    for line in (r.stdout or "").splitlines():
        print("   " + line)
    if r.stderr and r.stderr.strip():
        print("   [stderr tail] " + r.stderr.strip()[-600:])
    return r.returncode


def startup_probes(n):
    print(f"\n== import-startup probes ({n} fresh processes, G: env) ==")
    codes = []
    for i in range(n):
        r = subprocess.run([PY, "-I", "-X", "utf8", "-c",
                            "import numpy, xgboost, sklearn, polars, pyarrow; print('ok')"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        codes.append(r.returncode)
        tail = (r.stdout or "").strip()[-80:] or (r.stderr or "").strip()[-160:]
        print(f"   probe {i + 1:>2}: exit={r.returncode} (0x{r.returncode & 0xFFFFFFFF:08X}) {tail}")
    print(f"   summary: {sum(1 for c in codes if c == 0)} of {n} ok; non-zero codes: {[hex(c & 0xFFFFFFFF) for c in codes if c != 0]}")


def main():
    print("=== verify_E1_5_repro.py ===")
    print("start local:", time.strftime("%Y-%m-%d %H:%M:%S"), "| host commit/limit/free RAM now:",
          tuple(round(x, 2) for x in system_commit()))
    results = {}
    for nth in (1, 1, 4):
        results.setdefault(nth, []).append(run_child(nth))
    print("\nchild exit codes:", {k: [hex(c & 0xFFFFFFFF) for c in v] for k, v in results.items()})
    startup_probes(10)
    print("host commit/limit/free RAM after:", tuple(round(x, 2) for x in system_commit()))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--child":
        sys.exit(child(int(sys.argv[2])))
    main()
