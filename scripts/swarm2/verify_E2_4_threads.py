"""Independent verifier for E2-4: 'LightGBM CPU slows sharply with more OpenMP threads under load'.

Parent mode (default), project interpreter only:
  "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -B -X utf8 scripts\\swarm2\\verify_E2_4_threads.py
Child mode (one fresh process per timed run, started by the parent):
  ... verify_E2_4_threads.py --child THREADS ROWS TREES

Workload is the claim's: rng seed 1, float32 features, 20 columns, binary target, LightGBM params
identical to scripts\\swarm2\\E2_diag_lgb_threads.py. The claim script is also re-run unmodified.
Writes only logs\\swarm2\\verify_E2_4.log. Does not touch .pixi, E:, or any existing output.
"""
import ctypes
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time

REPO = r"G:\pureskill\csds-cs2"
SELF = os.path.abspath(__file__)
CLAIM_SCRIPT = os.path.join(REPO, "scripts", "swarm2", "E2_diag_lgb_threads.py")
LOG = os.path.join(REPO, "logs", "swarm2", "verify_E2_4.log")
FLAGS = ["-I", "-B", "-X", "utf8"]
THREADS = [1, 2, 4, 8, 16]
BLOCKS = 3
SLOW_RATIO = 5.0  # ASSUMPTION A1: "slow" means train time >= 5x the 1-thread time
OMP_WORDS = ("omp", "gomp", "iomp", "openblas", "mkl", "tbb", "lightgbm", "xgboost", "torch")


class _FT(ctypes.Structure):
    _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]


def sys_times():
    k32 = ctypes.WinDLL("kernel32")
    idle, kern, user = _FT(), _FT(), _FT()
    k32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))

    def val(f):
        return (f.hi << 32) | f.lo

    return val(idle), val(kern) + val(user)


def busy_between(a, b):
    d_idle = b[0] - a[0]
    d_total = b[1] - a[1]
    return 1.0 - d_idle / d_total if d_total > 0 else float("nan")


def loaded_runtime_dlls():
    psapi = ctypes.WinDLL("psapi")
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    h = ctypes.c_void_p(k32.GetCurrentProcess())
    arr = (ctypes.c_void_p * 4096)()
    need = ctypes.c_ulong(0)
    psapi.EnumProcessModules(h, arr, ctypes.sizeof(arr), ctypes.byref(need))
    cnt = min(need.value // ctypes.sizeof(ctypes.c_void_p), 4096)
    buf = ctypes.create_unicode_buffer(1024)
    names = []
    for i in range(cnt):
        if psapi.GetModuleFileNameExW(h, ctypes.c_void_p(arr[i]), buf, 1024):
            base = os.path.basename(buf.value)
            if any(w in base.lower() for w in OMP_WORDS):
                names.append(base)
    return sorted(set(names))


def child(threads, rows, trees):
    import numpy as np
    import lightgbm as lgb

    rng = np.random.default_rng(1)
    n, p = rows, 20
    X = rng.standard_normal((n, p), dtype=np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n) > 0).astype(np.int32)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, max_bin=255,
                  seed=0, verbose=-1, device_type="cpu", num_threads=threads)
    b0 = time.perf_counter()
    c0 = time.process_time()
    ds = lgb.Dataset(X, label=y, params=params).construct()
    b1 = time.perf_counter()
    c1 = time.process_time()
    s0 = sys_times()
    t0 = time.perf_counter()
    c2 = time.process_time()
    lgb.train(params, ds, num_boost_round=trees)
    t1 = time.perf_counter()
    c3 = time.process_time()
    s1 = sys_times()
    out = {"threads": threads, "rows": rows, "trees": trees, "lgb": lgb.__version__,
           "build_s": b1 - b0, "train_s": t1 - t0,
           "proc_cpu_build_s": c1 - c0, "proc_cpu_train_s": c3 - c2,
           "sys_busy_train": busy_between(s0, s1),
           "runtime_dlls": loaded_runtime_dlls()}
    print("CHILD_JSON " + json.dumps(out), flush=True)


def py_count():
    p = subprocess.run(["tasklist", "/fi", "imagename eq python.exe", "/fo", "csv", "/nh"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    return sum(1 for ln in p.stdout.splitlines() if ln.lower().startswith('"python.exe"'))


def med(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def run_child(threads, rows, trees, timeout):
    cmd = [PY] + FLAGS + [SELF, "--child", str(threads), str(rows), str(trees)]
    t0 = time.perf_counter()
    rc, out, err = None, "", ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=REPO)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    wall = time.perf_counter() - t0
    js = None
    for line in out.splitlines():
        if line.startswith("CHILD_JSON "):
            js = json.loads(line[len("CHILD_JSON "):])
    return {"rc": rc, "wall": wall, "js": js, "err_tail": (err or "")[-300:]}


def run_claim_script(timeout):
    cmd = [PY] + FLAGS + [CLAIM_SCRIPT]
    a = sys_times()
    t0 = time.perf_counter()
    rc, out = None, ""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=REPO)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        rc = "TIMEOUT"
    wall = time.perf_counter() - t0
    b = sys_times()
    return rc, out, wall, busy_between(a, b)


def parent():
    global PY
    PY = sys.executable
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    fh = open(LOG, "w", encoding="utf-8", newline="\n")

    def log(msg=""):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    try:
        is_project = os.path.normcase(PY).endswith(os.path.normcase(r".pixi\envs\default\python.exe"))
        log("VERIFY E2-4 (independent verifier, scripts\\swarm2\\verify_E2_4_threads.py)")
        log("Claim under test (E2-4, medium): 'LightGBM CPU slows sharply with more OpenMP threads under load.'")
        log("Cited evidence: E2_diag_lgb_threads.log, 20000 x 20, 20 trees, one process, order 1 -> 2 -> 4:"
            " 0.586 s, 2.039 s, 17.004 s (about 29x). E2_smoke_lgb_cpu.log: 16 threads 70.263 s."
            " E2_smoke4_lgb_cpu.log: 2000 rows, 3 trees, 4 threads, exit 124 at 100 s.")
        log("Proposed fix: num_threads=1 for CPU baselines while loaded; recheck thread scaling after the swarm.")
        log(f"interpreter: {PY}")
        log(f"project interpreter (G:\\pureskill\\csds-cs2\\.pixi\\envs\\default): {is_project}")
        log(f"os.cpu_count()={os.cpu_count()} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
            "(parent env; children run with -I)")
        log(f"python.exe processes at start: {py_count()}")
        a = sys_times()
        time.sleep(2)
        b = sys_times()
        log(f"system busy over 2 s at start: {busy_between(a, b):.3f}")
        log("ASSUMPTION A1: 'slow' = train time >= 5x the 1-thread time (about 1/6 of the claimed 29x). Pre-declared.")
        log("ASSUMPTION A2: every timed run is a fresh process (the claim ran 1 -> 2 -> 4 in one process).")
        log("ASSUMPTION A3: 3 blocks, thread order randomized per block (seed 2026+block). Host is NOT idle.")
        log("ASSUMPTION A4: 'load' = whole-system busy % over the LightGBM train window (GetSystemTimes).")
        log("")

        # Section 1: the claim script itself, unmodified, twice.
        log("== 1. Claim script re-run unmodified (E2_diag_lgb_threads.py), 2 reps")
        pat = re.compile(r"num_threads=(\d+) build=([\d.]+)s train_20_trees=([\d.]+)s")
        claim_ratio4 = []
        for rep in range(2):
            rc, out, wall, busy = run_claim_script(timeout=300)
            log(f"claim rep={rep} rc={rc} wall={wall:.2f}s sys_busy={busy:.3f}")
            times = {}
            for ln in out.splitlines():
                if ln.strip():
                    log(f"    {ln.strip()}")
                m = pat.search(ln)
                if m:
                    times[int(m.group(1))] = float(m.group(3))
            if 1 in times and 4 in times:
                claim_ratio4.append(times[4] / times[1])
                log(f"    ratio train(4)/train(1) = {times[4] / times[1]:.2f}")
        log("")

        # Section 2: randomized, interleaved matrix, fresh process per timed run.
        log("== 2. Matrix: 20000 x 20, 20 trees, threads 1/2/4/8/16, 3 randomized blocks (timeout 150 s)")
        results = []
        for blk in range(BLOCKS):
            order = THREADS[:]
            random.Random(2026 + blk).shuffle(order)
            log(f"-- block {blk} order {order}")
            for t in order:
                r = run_child(t, 20000, 20, timeout=150)
                js = r["js"]
                if js is None:
                    log(f"block={blk} threads={t:<2} rc={r['rc']} wall={r['wall']:.2f}s NO RESULT "
                        f"err_tail={r['err_tail']!r}")
                    results.append({"block": blk, "threads": t, "ok": False, "rc": r["rc"]})
                    continue
                eff = js["proc_cpu_train_s"] / js["train_s"] if js["train_s"] > 0 else float("nan")
                log(f"block={blk} threads={t:<2} rc={r['rc']} wall={r['wall']:.2f}s build={js['build_s']:.3f}s "
                    f"train={js['train_s']:.3f}s proc_cpu_train={js['proc_cpu_train_s']:.2f}s "
                    f"eff_par={eff:.2f} sys_busy_train={js['sys_busy_train']:.3f} lgb={js['lgb']}")
                if blk == 0 and t == order[0]:
                    log(f"    runtime DLLs in child (first run): {js['runtime_dlls']}")
                results.append({"block": blk, "threads": t, "ok": True, "train_s": js["train_s"],
                                "eff_par": eff, "busy": js["sys_busy_train"], "rc": r["rc"]})
        log("")

        # Section 3: the claim's 'did not finish in 100 s' case.
        log("== 3. Hang check: 2000 rows, 3 trees (claim: 4 threads did not finish in 100 s; timeout here 60 s)")
        for t in (4, 1):
            r = run_child(t, 2000, 3, timeout=60)
            js = r["js"]
            if js is None:
                log(f"hang-check threads={t} rc={r['rc']} wall={r['wall']:.2f}s NO RESULT")
            else:
                log(f"hang-check threads={t} rc={r['rc']} wall={r['wall']:.2f}s train={js['train_s']:.4f}s "
                    f"proc_cpu_train={js['proc_cpu_train_s']:.3f}s sys_busy_train={js['sys_busy_train']:.3f}")
        log("")

        # Section 4: summary and verdict.
        log("== 4. Summary")
        for t in THREADS:
            ok = [r for r in results if r["ok"] and r["threads"] == t]
            xs = [r["train_s"] for r in ok]
            if xs:
                log(f"threads={t:<2} n={len(xs)} median_train={med(xs):.3f}s min={min(xs):.3f}s max={max(xs):.3f}s "
                    f"median_eff_par={med([r['eff_par'] for r in ok]):.2f} "
                    f"median_sys_busy={med([r['busy'] for r in ok]):.3f}")
            else:
                log(f"threads={t:<2} n=0 (no valid runs)")
        log("")
        ratio = {}
        for blk in range(BLOCKS):
            base = [r for r in results if r["ok"] and r["block"] == blk and r["threads"] == 1]
            ratio[blk] = {}
            for t in (2, 4, 8, 16):
                cur = [r for r in results if r["ok"] and r["block"] == blk and r["threads"] == t]
                if base and cur:
                    ratio[blk][t] = cur[0]["train_s"] / base[0]["train_s"]
        log("paired ratio train(t)/train(1) within block:")
        for blk in range(BLOCKS):
            log(f"  block {blk}: " + "  ".join(f"t{t}={ratio[blk].get(t, float('nan')):.2f}" for t in (2, 4, 8, 16)))
        for t in (2, 4, 8, 16):
            vals = [ratio[b][t] for b in range(BLOCKS) if t in ratio[b]]
            log(f"  median over blocks t{t}: {med(vals):.2f}")
        log(f"  claim (E2_diag, one run): t2=3.48 t4=29.02; claim smoke: t16 = 70.263/0.586 = 119.9")
        log(f"  claim script re-run, t4/t1: {[round(x, 2) for x in claim_ratio4]}")
        pts = [(math.log(r["train_s"]), r["busy"]) for r in results if r["ok"] and r["threads"] > 1]
        try:
            log(f"corr(log train_s, sys_busy_train) over multi-thread runs (n={len(pts)}): "
                f"{statistics.correlation([p[0] for p in pts], [p[1] for p in pts]):.3f}")
        except (statistics.StatisticsError, ValueError) as exc:
            log(f"corr(log train_s, sys_busy_train) not computed: {exc!r}")
        pts2 = [(r["eff_par"], r["threads"]) for r in results if r["ok"] and r["threads"] > 1]
        try:
            log(f"corr(eff_par, threads) over multi-thread runs (n={len(pts2)}): "
                f"{statistics.correlation([p[0] for p in pts2], [p[1] for p in pts2]):.3f}")
        except (statistics.StatisticsError, ValueError) as exc:
            log(f"corr(eff_par, threads) not computed: {exc!r}")
        log("")
        n_slow = sum(1 for b in range(BLOCKS) if 4 in ratio[b] and ratio[b][4] >= SLOW_RATIO)
        n_fast = sum(1 for b in range(BLOCKS) if 4 in ratio[b] and ratio[b][4] < 2.0)
        if n_slow >= 2:
            verdict = "REPRODUCED this session (t4/t1 >= 5 in >= 2 of 3 blocks)"
        elif n_fast >= 2:
            verdict = "NOT REPRODUCED this session (t4/t1 < 2 in >= 2 of 3 blocks)"
        else:
            verdict = "MIXED this session"
        log(f"verdict on the thread-scaling numbers (pre-declared rule): {verdict}")
        log("Scope: this measures the claim's workload on the project env, on a host that was not idle.")
        log("Not run: the claim's own interpreter (cs2gpu_alt, per E2.md), an idle-host rerun, or non-synthetic CS2 data.")
        log("[done]")
    finally:
        fh.close()


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        child(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        parent()
