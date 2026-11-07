r"""Independent verification of claimed finding E2-4 (LightGBM CPU thread scaling).

Claim under test: LightGBM CPU training gets sharply slower as num_threads rises under load.
Cited evidence (logs\swarm2\E2_diag_lgb_threads.log): 20k x 20 binary data, 20 trees, 63 leaves,
one run per thread count in the order 1 -> 2 -> 4: 0.586 s, 2.039 s, 17.004 s (about 29x).
Also cited: 16 threads 70.263 s (smoke, not in the log), and 2000 rows / 3 trees / 4 threads did
not finish in 100 s (exit 124).

Run with the project interpreter only:
  "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 verify_E2_4_sweep.py --mode all

All output is printed and appended to LOG_PATH (logs\swarm2\verify_E2_4.log).

Modes:
  sweep   Thread sweep on the claimed workload. Each run is a fresh interpreter. Run order is
          shuffled within each block so ambient-load drift is shared across thread counts.
          Records ambient system CPU busy cores for the 1 s window before each run, and the
          process CPU time spent in training (CPU/wall shows whether threads spin or wait).
  small   2000 rows, 3 trees, threads 1/2/4, one run each, 100 s timeout (the claimed hang).
  loaded  Same workload for threads 1 and 4 while --busy pure-Python busy processes run
          (one core each, ASSUMPTION) to test the oversubscription mechanism directly.
  worker  Internal: one configuration in a fresh interpreter, prints one JSON line.
"""
import argparse
import ctypes
import json
import os
import random
import statistics
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime
from importlib import metadata

SELF = os.path.abspath(__file__)
BUSY = os.path.join(os.path.dirname(SELF), "verify_E2_4_busy.py")
LOG_PATH = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E2_4.log"
NCPU = os.cpu_count() or 16
WORKLOAD = dict(n=20000, p=20, trees=20)  # the claimed workload (num_leaves=63, max_bin=255, lr=0.1)
SEED = 20261013  # ASSUMPTION: fixed seed for run-order shuffling only
RUN_TIMEOUT = 150  # ASSUMPTION: seconds per worker before it is counted as a timeout


class FILETIME(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


def emit(msg=""):
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")


def system_times():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    if not ok:
        raise OSError("GetSystemTimes failed")

    def as_int(ft):
        return (ft.hi << 32) | ft.lo

    # On Windows the kernel time already includes idle time.
    return as_int(idle), as_int(kern), as_int(user)


def busy_cores(t0, t1):
    idle = t1[0] - t0[0]
    total = (t1[1] - t0[1]) + (t1[2] - t0[2])
    if total <= 0:
        return float("nan")
    return NCPU * (total - idle) / total


def worker(args):
    import lightgbm as lgb
    import numpy as np

    rng = np.random.default_rng(1)
    X = rng.standard_normal((args.n, args.p), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(args.n) > 0).astype(np.int32)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                  seed=0, verbose=-1, device_type="cpu", num_threads=args.threads)
    c0, w0 = time.process_time(), time.perf_counter()
    ds = lgb.Dataset(X, label=y, params=params).construct()
    build = {"wall": time.perf_counter() - w0, "cpu": time.process_time() - c0}
    runs = []
    for _ in range(args.reps):
        c0, w0 = time.process_time(), time.perf_counter()
        lgb.train(params, ds, num_boost_round=args.trees)
        runs.append({"wall": time.perf_counter() - w0, "cpu": time.process_time() - c0})
    out = {"threads": args.threads, "n": args.n, "trees": args.trees, "reps": args.reps,
           "lgb": lgb.__version__, "build": build, "runs": runs}
    print(json.dumps(out), flush=True)


def run_one(block, threads, n, p, trees, reps, timeout):
    cmd = [sys.executable, "-I", "-X", "utf8", SELF, "--mode", "worker",
           "--threads", str(threads), "--n", str(n), "--p", str(p),
           "--trees", str(trees), "--reps", str(reps)]
    a0 = system_times()
    time.sleep(1.0)
    a1 = system_times()
    ambient = busy_cores(a0, a1)
    t0 = time.perf_counter()
    status, res = "ok", None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            status = f"rc={proc.returncode}"
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                res = json.loads(line)
        if status == "ok" and res is None:
            status = "no_json"
    except subprocess.TimeoutExpired:
        status = "timeout"
    return dict(block=block, threads=threads, status=status, ambient=ambient,
                elapsed=time.perf_counter() - t0, res=res)


def describe(rec):
    head = (f"block={rec['block']} threads={rec['threads']:>2} status={rec['status']} "
            f"ambient_busy_cores={rec['ambient']:.2f} free_cores={NCPU - rec['ambient']:.2f} "
            f"elapsed_total={rec['elapsed']:.1f}s")
    if rec["res"] is None:
        return head
    runs = " ".join(f"[wall={x['wall']:.3f}s cpu={x['cpu']:.3f}s]" for x in rec["res"]["runs"])
    return head + f" build_wall={rec['res']['build']['wall']:.3f}s " + runs


def build_schedule(plan, seed):
    """plan: list of (threads, n_blocks, reps_per_run). Each block holds each eligible thread
    count once, in a shuffled order, so that drift in ambient load is shared across counts."""
    rng = random.Random(seed)
    schedule = []
    n_blocks = max(b for _, b, _ in plan)
    for b in range(n_blocks):
        group = [(t, reps) for t, nb, reps in plan if b < nb]
        rng.shuffle(group)
        schedule.extend((b, t, reps) for t, reps in group)
    return schedule


def summarize(records, label):
    emit(f"--- summary ({label}) ---")
    by_t = {}
    for rec in records:
        by_t.setdefault(rec["threads"], []).append(rec)
    for t in sorted(by_t):
        recs = by_t[t]
        ok = [r for r in recs if r["res"] is not None]
        n_to = sum(r["status"] == "timeout" for r in recs)
        amb = [r["ambient"] for r in recs]
        if ok:
            cold = [r["res"]["runs"][0] for r in ok]
            walls = [x["wall"] for x in cold]
            cpu_ratio = [x["cpu"] / x["wall"] for x in cold]
            emit(f"threads={t:>2} n_runs={len(recs)} completed={len(ok)} timeouts={n_to} "
                 f"cold_train_wall median={statistics.median(walls):.3f}s "
                 f"min={min(walls):.3f}s max={max(walls):.3f}s "
                 f"cpu_over_wall_median={statistics.median(cpu_ratio):.2f} "
                 f"ambient_busy_cores_median={statistics.median(amb):.2f}")
        else:
            emit(f"threads={t:>2} n_runs={len(recs)} completed=0 timeouts={n_to} "
                 f"ambient_busy_cores_median={statistics.median(amb):.2f} "
                 f"(no completed run; timeouts mean wall time above the timeout)")
    base = {r["block"]: r for r in records if r["threads"] == 1 and r["res"] is not None}
    for t in sorted(by_t):
        if t == 1:
            continue
        pairs = []
        for r in by_t[t]:
            b = base.get(r["block"])
            if r["res"] is not None and b is not None:
                pairs.append(r["res"]["runs"][0]["wall"] / b["res"]["runs"][0]["wall"])
        if pairs:
            emit(f"paired cold ratio threads={t} / threads=1, same block: "
                 + ", ".join(f"{x:.2f}" for x in pairs)
                 + f" | median={statistics.median(pairs):.2f}")
    for t in sorted(by_t):
        warm = [r["res"]["runs"][1]["wall"] for r in by_t[t]
                if r["res"] is not None and len(r["res"]["runs"]) > 1]
        if warm:
            emit(f"warm train (2nd call, same process) threads={t}: "
                 f"median={statistics.median(warm):.3f}s n={len(warm)}")


def mode_sweep():
    emit("=== mode sweep: claimed workload, fresh interpreter per run ===")
    emit(f"workload={WORKLOAD} num_leaves=63 max_bin=255 learning_rate=0.1 ncpu={NCPU} "
         f"timeout={RUN_TIMEOUT}s")
    plan = [(1, 5, 2), (2, 5, 2), (4, 5, 2), (8, 2, 1), (16, 2, 1)]
    records = []
    for block, t, reps in build_schedule(plan, SEED):
        rec = run_one(block, t, WORKLOAD["n"], WORKLOAD["p"], WORKLOAD["trees"], reps, RUN_TIMEOUT)
        records.append(rec)
        emit(describe(rec))
    summarize(records, "sweep")
    return records


def mode_small():
    emit("=== mode small: 2000 rows, 3 trees, one run each, 100 s timeout ===")
    for t in (1, 2, 4):
        rec = run_one(0, t, 2000, 20, 3, 1, 100)
        emit(describe(rec))


def mode_loaded(busy_n):
    emit(f"=== mode loaded: {busy_n} busy-loop processes, one core each (ASSUMPTION) ===")
    procs = []
    try:
        for _ in range(busy_n):
            procs.append(subprocess.Popen([sys.executable, "-I", "-X", "utf8", BUSY, "600"]))
        time.sleep(4.0)
        a0 = system_times()
        time.sleep(3.0)
        a1 = system_times()
        emit(f"busy processes alive={sum(p.poll() is None for p in procs)} "
             f"ambient_busy_cores_with_load={busy_cores(a0, a1):.2f} of {NCPU}")
        records = []
        for block, t, reps in build_schedule([(1, 3, 2), (4, 3, 2)], SEED + 1):
            rec = run_one(block, t, WORKLOAD["n"], WORKLOAD["p"], WORKLOAD["trees"], reps, RUN_TIMEOUT)
            records.append(rec)
            emit(describe(rec))
        summarize(records, f"loaded busy={busy_n}")
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in procs:
            p.wait()
        emit("busy processes stopped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "sweep", "small", "loaded", "worker"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n", type=int, default=WORKLOAD["n"])
    ap.add_argument("--p", type=int, default=WORKLOAD["p"])
    ap.add_argument("--trees", type=int, default=WORKLOAD["trees"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--busy", type=int, default=12)
    args = ap.parse_args()
    if args.mode == "worker":
        worker(args)
        return
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    emit("")
    emit(f"##### verify_E2_4_sweep.py run {datetime.now().isoformat(timespec='seconds')}")
    emit(f"driver python={sys.version.split()[0]} lightgbm={metadata.version('lightgbm')} "
         f"ncpu={NCPU}")
    emit("claimed (E2 log): 1 thread 0.586 s, 2 threads 2.039 s, 4 threads 17.004 s")
    emit(f"claimed ratio 4/1 = 17.004/0.586 = {17.004 / 0.586:.2f}; "
         f"2/1 = 2.039/0.586 = {2.039 / 0.586:.2f}")
    if args.mode in ("all", "sweep"):
        mode_sweep()
    if args.mode in ("all", "small"):
        mode_small()
    if args.mode in ("all", "loaded"):
        mode_loaded(args.busy)
    emit("##### done")


if __name__ == "__main__":
    main()
