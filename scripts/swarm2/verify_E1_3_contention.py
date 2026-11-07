"""Independent verifier for E1-3 (contended timings), see notes\\swarm2\\E1.md.

Mandated interpreter (run this way):
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_E1_3_contention.py

This env has xgboost 3.4.2 built WITHOUT CUDA, so the CUDA figures cannot be re-measured here.
What this script does (workload parameters are ASSUMPTIONS, copied from scripts\\swarm2\\E1_bench.py):
  1. Recomputes the arithmetic of the claimed numbers from the raw run times in the E1 logs.
  2. Interleaves CPU default threads (16 logical CPUs) and nthread=1 on the same
     1,000,000 x 50 float32 workload, 10 trees per timed run, 4 paired reps (order alternates).
  3. Per timed run: wall time, process CPU time (spinning vs descheduled threads),
     system busy logical CPUs (GetSystemTimes), GPU util and memory before and after (nvidia-smi).
  4. Checks whether the CPU thread count changes predictions (same seed, same trees).
  5. Confirms the CUDA device is unavailable in this env.
Writes only logs\\swarm2\\verify_E1_3.log.
"""
import ctypes
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = r"G:\pureskill\csds-cs2"
LOG = os.path.join(ROOT, "logs", "swarm2", "verify_E1_3.log")

N_TRAIN = 1_000_000  # ASSUMPTION: same as E1_bench.py
N_HOLD = 100_000  # ASSUMPTION: same generator as E1_bench.py (holdout slice unused here)
N_FEAT = 50  # ASSUMPTION
SEED = 12345  # ASSUMPTION: same as E1_bench.py
MAX_DEPTH = 4  # ASSUMPTION: same as E1_bench.py
ETA = 0.1  # ASSUMPTION: same as E1_bench.py
TREES_TIMED = 10  # ASSUMPTION: reduced from 200 so the run stays bounded under load
WARM_TREES = 2  # ASSUMPTION
REPS = 4  # ASSUMPTION: paired reps, order alternates
N_CHECK = 100_000  # ASSUMPTION: rows used for the prediction-equality check

NCPU = os.cpu_count() or 16
_LOG_LINES = []


def out(msg=""):
    print(msg, flush=True)
    _LOG_LINES.append(msg)


def flush_log():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG_LINES) + "\n")


# ---------- Windows system counters (no extra packages) ----------
class FILETIME(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]


_k32 = ctypes.WinDLL("kernel32")


def _ft(f):
    return (f.hi << 32) | f.lo


def sys_times():
    """Cumulative (idle, kernel incl. idle, user) across all logical CPUs, 100 ns units."""
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    if not _k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")
    return _ft(idle), _ft(kern), _ft(user)


def busy_cores(s0, s1):
    """Average number of busy logical CPUs over the window [s0, s1]."""
    tot = (s1[1] - s0[1]) + (s1[2] - s0[2])
    idle = s1[0] - s0[0]
    if tot <= 0:
        return float("nan")
    return (tot - idle) / tot * NCPU


def gpu_state():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip().replace("\n", " | ")
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {type(e).__name__}"


def top_procs(tag):
    cmd = ("Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 ProcessName,Id,"
           "@{n='CPU_s';e={[math]::Round($_.CPU,1)}},@{n='WS_MB';e={[math]::Round($_.WorkingSet64/1MB,0)}} "
           "| Format-Table -AutoSize | Out-String -Width 200")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=120)
        out(f"[top processes by cumulative CPU_s, {tag}]")
        for line in r.stdout.splitlines():
            if line.strip():
                out("  " + line.rstrip())
    except Exception as e:  # noqa: BLE001
        out(f"[top processes, {tag}] failed: {type(e).__name__}")


def free_ram_gb():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                           capture_output=True, text=True, timeout=60)
        return round(int(r.stdout.strip()) / 2**20, 2)
    except Exception:  # noqa: BLE001
        return float("nan")


# ---------- workload (same generator as E1_bench.py make_data) ----------
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
    return X[:N_TRAIN], y[:N_TRAIN]


def params(nthread):
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": MAX_DEPTH,
        "eta": ETA,
        "seed": 0,
    }
    if nthread is not None:
        p["nthread"] = nthread
    return p


# ---------- section 1: arithmetic of the claimed numbers ----------
def arithmetic():
    out("== 1. Arithmetic of the claimed numbers ==")
    out("Inputs copied from logs\\swarm2\\E1_bench.log, E1_diag.log, E1_threads.log (verbatim values).")
    trees = 200
    cpu_def_total = 399.585  # E1_bench.log: [cpu_default] run 1/1
    cpu_1t_runs = [80.057, 81.517]  # E1_bench.log: [cpu_1thread] runs
    cuda_runs = [6.302, 4.122, 4.590]  # E1_bench.log: [cuda] runs
    cpu_1t_med = statistics.median(cpu_1t_runs)
    cuda_med = statistics.median(cuda_runs)
    out(f"cpu_default s/tree            = {cpu_def_total / trees:.4f}   (claim 1.998; 1 run only)")
    out(f"cpu_1thread median / s/tree   = {cpu_1t_med:.3f} / {cpu_1t_med / trees:.4f}   (claim 80.787 / 0.404)")
    out(f"cuda median / s/tree          = {cuda_med:.3f} / {cuda_med / trees:.5f}   (claim 4.590 / 0.023)")
    out(f"speedup cuda vs cpu_default   = {cpu_def_total / cuda_med:.2f}x   (claim 87.1x; denominator is a single run)")
    out(f"speedup cuda vs cpu_1thread   = {cpu_1t_med / cuda_med:.2f}x   (claim 17.6x)")
    out(f"cpu_default vs cpu_1thread    = {cpu_def_total / cpu_1t_med:.2f}x slower per tree with default threads (same session)")
    out(f"cuda run spread max/min       = {max(cuda_runs) / min(cuda_runs):.2f}x (3 runs); "
        f"cpu_1thread spread = {max(cpu_1t_runs) / min(cpu_1t_runs):.3f}x (2 runs)")
    diag_def = 73.10 / 20
    out(f"E1_diag same config (1M, 20 trees, no warm-up) cpu_default = {diag_def:.3f} s/tree = "
        f"{diag_def / (cpu_def_total / trees):.2f}x the bench's 1.998 s/tree")
    out(f"E1_diag nthread=4 (1M, 20 trees) = {23.50 / 20:.3f} s/tree")
    out(f"E1_diag cuda (1M, 20 trees, first CUDA call in process, includes context init) = {14.63 / 20:.3f} s/tree; "
        f"not comparable to bench median")
    out(f"E1_threads 100k, 10 trees: nthread=1 {0.502 / 10:.4f} s/tree vs nthread=2 {14.304 / 10:.4f} s/tree "
        f"= {14.304 / 0.502:.1f}x slower with 2 threads")
    out(f"E1_threads nthread=16 runs were 22.472 s and 0.739 s (30x apart, same config)")
    out("")


# ---------- section 2: live CPU measurements ----------
def timed(name, nt, dtrain, rep, keep):
    s0 = sys_times()
    g0 = gpu_state()
    c0 = time.process_time()
    t0 = time.perf_counter()
    booster = xgb.train(params(nt), dtrain, num_boost_round=TREES_TIMED)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    s1 = sys_times()
    g1 = gpu_state()
    if rep == 0:
        keep[name] = booster
    sysb = busy_cores(s0, s1)
    self_cores = cpu / wall
    rec = {
        "name": name, "rep": rep, "wall": wall, "s_per_tree": wall / TREES_TIMED,
        "self_cores": self_cores, "sys_busy": sysb, "other_busy": sysb - self_cores,
        "gpu0": g0, "gpu1": g1,
    }
    out(f"[{name} rep {rep}] wall={wall:.3f} s  s/tree={wall / TREES_TIMED:.4f}  "
        f"process_cpu/wall={self_cores:.2f} cores  system_busy={sysb:.2f} of {NCPU} logical  "
        f"other_busy~{sysb - self_cores:.2f}  gpu_before[util%,MiB]={g0}  gpu_after={g1}")
    return rec


def live(dtrain, dcheck_X):
    out("== 2. Live measurements on this host (G: env, CPU only) ==")
    configs = [("cpu_1thread", 1), ("cpu_default16", None)]
    keep = {}
    for name, nt in configs:
        t0 = time.perf_counter()
        xgb.train(params(nt), dtrain, num_boost_round=WARM_TREES)
        out(f"warm-up {name}: {WARM_TREES} trees in {time.perf_counter() - t0:.3f} s (untimed)")
    runs = []
    for rep in range(REPS):
        order = configs if rep % 2 == 0 else list(reversed(configs))
        for name, nt in order:
            runs.append(timed(name, nt, dtrain, rep, keep))

    out("")
    out("summary per config (s/tree, median of runs; load columns are means over runs):")
    for name, _ in configs:
        rs = [r for r in runs if r["name"] == name]
        spt = [r["s_per_tree"] for r in rs]
        out(f"  {name:<14} runs s/tree={[round(x, 4) for x in spt]}  median={statistics.median(spt):.4f}  "
            f"process_cores={statistics.mean(r['self_cores'] for r in rs):.2f}  "
            f"other_busy={statistics.mean(r['other_busy'] for r in rs):.2f}  "
            f"system_busy={statistics.mean(r['sys_busy'] for r in rs):.2f}/{NCPU}")

    pairs = []
    for rep in range(REPS):
        a = next(r for r in runs if r["name"] == "cpu_1thread" and r["rep"] == rep)
        b = next(r for r in runs if r["name"] == "cpu_default16" and r["rep"] == rep)
        pairs.append(b["wall"] / a["wall"])
    out(f"paired ratio default16 / 1thread per rep = {[round(x, 2) for x in pairs]}  "
        f"median={statistics.median(pairs):.2f}  (claim per-tree ratio 4.95)")
    out(f"reps where default16 slower than 1thread: {sum(x > 1 for x in pairs)} of {len(pairs)}")
    med_def = statistics.median([r["s_per_tree"] for r in runs if r["name"] == "cpu_default16"])
    med_1t = statistics.median([r["s_per_tree"] for r in runs if r["name"] == "cpu_1thread"])
    out(f"live default16 / 1thread median s/tree ratio = {med_def / med_1t:.2f}")
    out("sensitivity (illustrative, CUDA NOT re-measured): if the CUDA median 0.02295 s/tree stayed fixed, "
        f"speedup vs live default16 = {med_def / 0.02295:.1f}x and vs live 1thread = {med_1t / 0.02295:.1f}x")
    out("")

    out("== 3. Prediction equality across thread counts (same seed, same trees) ==")
    dcheck = xgb.DMatrix(dcheck_X)
    p1 = keep["cpu_1thread"].predict(dcheck)
    p16 = keep["cpu_default16"].predict(dcheck)
    d = np.abs(p1.astype(np.float64) - p16.astype(np.float64))
    out(f"cpu_1thread vs cpu_default16 on {len(dcheck_X)} rows: max_abs={d.max():.3e} mean_abs={d.mean():.3e}  "
        f"identical={bool(np.array_equal(p1, p16))}")
    out("")
    return runs


# ---------- section 4: CUDA availability in this env ----------
def cuda_check(X, y):
    out("== 4. CUDA availability in the mandated env ==")
    out(f"xgboost {xgb.__version__} build_info USE_CUDA = {xgb.build_info().get('USE_CUDA')}")
    small = xgb.DMatrix(X[:2000], label=y[:2000])
    try:
        xgb.train({"tree_method": "hist", "device": "cuda", "max_depth": 2}, small, num_boost_round=2)
        out("device=cuda trained (unexpected for this env)")
    except Exception as e:  # noqa: BLE001
        out(f"device=cuda failed in this env: {type(e).__name__}: {str(e)[:200]}")
    out("")


def run():
    out("verify_E1_3 (independent check of contended CPU/GPU timings)")
    out(f"python {sys.version.split()[0]} {sys.executable}")
    out(f"xgboost {xgb.__version__} logical_cpus={NCPU} free_ram_GB={free_ram_gb()}")
    out(f"workload ASSUMPTIONS: n_train={N_TRAIN} n_feat={N_FEAT} depth={MAX_DEPTH} eta={ETA} "
        f"timed_trees={TREES_TIMED} warm_trees={WARM_TREES} reps={REPS} seed={SEED}")
    out(f"start gpu[util%,MiB]={gpu_state()}")
    top_procs("start")
    out("")

    arithmetic()

    t0 = time.perf_counter()
    X, y = make_data()
    out(f"data generated in {time.perf_counter() - t0:.2f} s; X_train {X.shape} {X.dtype}")
    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(X, label=y)
    out(f"DMatrix build {time.perf_counter() - t0:.3f} s")
    out("")

    live(dtrain, X[:N_CHECK])
    cuda_check(X, y)

    out(f"end gpu[util%,MiB]={gpu_state()}")
    top_procs("end")


def main():
    try:
        run()
    finally:
        flush_log()


if __name__ == "__main__":
    main()
