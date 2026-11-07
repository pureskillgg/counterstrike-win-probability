"""verify_E1_4 driver: runs the recomputation experiments and writes logs\\swarm2\\verify_E1_4.log.

Experiments (each is a separate subprocess run of verify_E1_4_bench.py):
  A  sweepG      : G: project interpreter, one process, 3 interleaved rounds over nthread 1,2,4,8,16
  B  childG      : G: project interpreter, one fresh process per nthread, 5 timed runs each, default env
  C  childG_pass : as B, with OMP_WAIT_POLICY=PASSIVE (tests the OpenMP spin hypothesis)
  D  sweepC      : C: cs2gpu interpreter (CUDA build, the env the E1 claim used), 3 interleaved rounds

Host context: typeperf logs total CPU %, processor queue length, context switches/s, pages/s and
available MB every 1 s to the scratchpad; per-process CPU deltas are taken from Get-Process before/after
each experiment. Raw JSON stays in the scratchpad; the summary goes to logs\\swarm2\\verify_E1_4.log.

Run:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E1_4_driver.py
"""
import json
import os
import statistics
import subprocess
import sys
import time

G_PY = r"G:\pureskill\csds-cs2\.pixi\envs\default\python.exe"
C_PY = r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu-15282324412032665829\envs\default\python.exe"
BENCH = r"G:\pureskill\csds-cs2\scripts\swarm2\verify_E1_4_bench.py"
LOG = r"G:\pureskill\csds-cs2\logs\swarm2\verify_E1_4.log"
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
NTHREADS = [1, 2, 4, 8, 16]
CHILD_REPS = 5
SWEEP_ROUNDS = 3


def log(text=""):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text, flush=True)


def proc_snapshot():
    cmd = ("Get-Process | Select-Object Id,Name,CPU | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                         timeout=120).stdout
    try:
        rows = json.loads(out)
    except json.JSONDecodeError:
        return {}
    if isinstance(rows, dict):
        rows = [rows]
    return {int(r["Id"]): (r["Name"], float(r["CPU"] or 0.0)) for r in rows}


def top_consumers(before, after, n=8):
    deltas = []
    for pid, (name, cpu1) in after.items():
        if pid in before:
            d = cpu1 - before[pid][1]
            if d > 0:
                deltas.append((d, pid, name))
    deltas.sort(reverse=True)
    return [{"pid": pid, "name": name, "cpu_s": round(d, 2)} for d, pid, name in deltas[:n]]


def run_bench(py, args, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    t0 = time.time()
    p = subprocess.run([py, "-B", "-I", "-X", "utf8", BENCH] + args, capture_output=True, text=True,
                       env=env, timeout=3600)
    t1 = time.time()
    js = [line for line in p.stdout.splitlines() if line.startswith("JSON ")]
    rec = json.loads(js[-1][5:]) if js else None
    return {"rec": rec, "rc": p.returncode, "stderr_tail": p.stderr[-1500:], "t0": t0, "t1": t1,
            "stdout_head": p.stdout[:400]}


def fmt(x, nd=4):
    return "nan" if x is None or x != x else f"{x:.{nd}f}"


def summarize_runs(runs, label):
    log(f"  summary [{label}] (s/tree = wall/10 trees; busy% = system busy over the run window)")
    log(f"  {'nthread':>7} {'n':>3} {'median s/tree':>14} {'min':>8} {'max':>8} {'IQR':>8} "
        f"{'x vs 1thr':>10} {'med cpu/wall':>13} {'med busy%':>10} {'med pf':>8}")
    base = None
    for nt in NTHREADS:
        rs = [r for r in runs if r["nthread"] == nt]
        if not rs:
            continue
        spt = [r["s_per_tree"] for r in rs]
        med = statistics.median(spt)
        if nt == 1:
            base = med
        q = statistics.quantiles(spt, n=4) if len(spt) >= 2 else [med, med, med]
        iqr = q[2] - q[0]
        ratio = (med / base) if base else float("nan")
        cw = statistics.median([r["cpu_over_wall"] for r in rs])
        busy = statistics.median([r["host_busy_window_pct"] for r in rs])
        pf = statistics.median([r["page_faults"] for r in rs])
        log(f"  {nt:>7} {len(rs):>3} {med:>14.4f} {min(spt):>8.4f} {max(spt):>8.4f} {iqr:>8.4f} "
            f"{ratio:>10.2f} {cw:>13.2f} {busy:>10.1f} {pf:>8.0f}")
    # Is one thread the fastest setting in this experiment?
    med_by_nt = {nt: statistics.median([r["s_per_tree"] for r in runs if r["nthread"] == nt])
                 for nt in NTHREADS if any(r["nthread"] == nt for r in runs)}
    best_nt = min(med_by_nt, key=med_by_nt.get)
    log(f"  fastest median setting: nthread={best_nt} ({med_by_nt[best_nt]:.4f} s/tree); "
        f"nthread=1 median {med_by_nt.get(1, float('nan')):.4f} s/tree")
    # Pooled correlation of log(s/tree) with host busy % for multi-thread settings
    pts = [(r["host_busy_window_pct"], r["s_per_tree"]) for r in runs if r["nthread"] > 1
           and r["host_busy_window_pct"] == r["host_busy_window_pct"]]
    if len(pts) >= 3:
        import math
        xs = [p[0] for p in pts]
        ys = [math.log(p[1]) for p in pts]
        try:
            r_val = statistics.correlation(xs, ys)
        except statistics.StatisticsError:
            r_val = float("nan")
        log(f"  pearson r(busy%, log s/tree) over nthread>1 runs: {fmt(r_val, 3)} (n={len(pts)})")


def run_sweep_experiment(label, py, tag):
    before = proc_snapshot()
    res = run_bench(py, ["sweep", "--rounds", str(SWEEP_ROUNDS), "--tag", tag])
    after = proc_snapshot()
    with open(os.path.join(SCRATCH, f"verify_E1_4_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log("")
    log(f"=== {label} ===  python={py}  rc={res['rc']}  wall={res['t1'] - res['t0']:.1f}s")
    if res["rec"] is None:
        log(f"  NO RESULT. stderr tail: {res['stderr_tail']}")
        return res
    rec = res["rec"]
    log(f"  xgboost={rec['xgboost']} use_cuda={rec['use_cuda']} rounds={rec['rounds']}")
    log(f"  OpenMP-related modules in process: {[os.path.basename(m) for m in rec['omp_modules']]}")
    log(f"  OMP/KMP env in child: {rec['env_omp']}")
    log(f"  top CPU consumers during experiment (cpu s): {top_consumers(before, after)}")
    summarize_runs(rec["runs"], label)
    for r in rec["runs"]:
        log(f"    round={r['rep']} nthread={r['nthread']:>2} s/tree={r['s_per_tree']:.4f} "
            f"cpu/wall={r['cpu_over_wall']} busy%={r['host_busy_window_pct']} pf={r['page_faults']}")
    return res


def run_child_experiment(label, py, tag, extra_env=None):
    before = proc_snapshot()
    runs = []
    omp_modules = None
    per_nt = {}
    for nt in NTHREADS:
        res = run_bench(py, ["child", "--nthread", str(nt), "--reps", str(CHILD_REPS), "--tag", tag],
                        extra_env=extra_env)
        per_nt[nt] = res
        if res["rec"] is not None:
            runs.extend(res["rec"]["runs"])
            omp_modules = omp_modules or res["rec"]["omp_modules"]
    after = proc_snapshot()
    with open(os.path.join(SCRATCH, f"verify_E1_4_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump({nt: {k: v for k, v in res.items() if k != "stdout_head"} for nt, res in per_nt.items()},
                  fh, indent=1)
    log("")
    log(f"=== {label} ===  python={py}  extra_env={extra_env}")
    log(f"  OpenMP-related modules in child: {[os.path.basename(m) for m in (omp_modules or [])]}")
    log(f"  top CPU consumers during experiment (cpu s): {top_consumers(before, after)}")
    if not runs:
        log("  NO RESULT from any child")
        return per_nt
    summarize_runs(runs, label)
    for r in runs:
        log(f"    nthread={r['nthread']:>2} rep={r['rep']} s/tree={r['s_per_tree']:.4f} "
            f"cpu/wall={r['cpu_over_wall']} busy%={r['host_busy_window_pct']} pf={r['page_faults']}")
    return per_nt


def start_typeperf():
    csv_path = os.path.join(SCRATCH, "verify_E1_4_host_counters.csv")
    counters = [r"\Processor(_Total)\% Processor Time", r"\System\Processor Queue Length",
                r"\System\Context Switches/sec", r"\Memory\Pages/sec", r"\Memory\Available MBytes"]
    p = subprocess.Popen(["typeperf"] + counters + ["-si", "1", "-o", csv_path, "-y"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, csv_path


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as fh:
        fh.write("verify_E1_4 log: independent recomputation of the E1-4 CPU thread-count sweep\n")
        fh.write(f"started {time.strftime('%Y-%m-%d %H:%M:%S')} (local)\n")
        fh.write("Workload (ASSUMPTION carried from E1_threads.py): 100,000 x 50 float32, seed 7, 10 trees, depth 4,\n"
                 "eta 0.1, hist, binary:logistic, 2 untimed warm-up trees per setting.\n")
        fh.write("Claimed E1-4 numbers (s/tree): nthread 1 0.050; 2 1.43; 4 1.76; 8 5.07; 16 1.16 "
                 "(2 runs, 22.47 s and 0.739 s).\n")
    tp, csv_path = start_typeperf()
    log(f"host counters -> {csv_path} (typeperf, 1 s)")
    try:
        run_sweep_experiment("A sweepG (G: project env, interleaved, in-process)", G_PY, "sweepG")
        run_child_experiment("B childG (G: project env, fresh process per nthread, default env)", G_PY, "childG")
        run_child_experiment("C childG_pass (as B with OMP_WAIT_POLICY=PASSIVE)", G_PY, "childG_pass",
                             extra_env={"OMP_WAIT_POLICY": "PASSIVE"})
        run_sweep_experiment("D sweepC (C: cs2gpu CUDA env, the env the E1 claim used)", C_PY, "sweepC")
    finally:
        tp.terminate()
        log("")
        log("done " + time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
