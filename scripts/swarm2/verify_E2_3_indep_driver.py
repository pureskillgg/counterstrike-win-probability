"""VERIFY E2-3, independent pass: ABBA driver for claim-size CatBoost CPU (1 thread) vs GPU fits.

Runs verify_E2_3_indep_bench.py as child processes in the order
  CPU, GPU, GPU, CPU, CPU, GPU, GPU, CPU
Each child does fits of 100, 100 and 5 trees on the same 100000 x 50 data (5-tree fit = overhead probe).
Then prints: per-fit table, medians and ratio, exact permutation test (8 vs 8, all C(16,8) splits),
per-block ratios (adjacent child pairs), and an overhead split (intercept and slope, ASSUMPTION linear in trees).

Usage (cs2gpu_alt interpreter):
  "C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe" -I -X utf8 verify_E2_3_indep_driver.py
Stdout is redirected by the caller to logs\\swarm2\\verify_E2_3.log (appended, after the other verifier's section).
"""
import itertools
import json
import statistics
import subprocess
import time

PY = r"C:\Users\<user>\.pixi\workspace-envs\cs2gpu_alt-16300978804608365045\envs\default\python.exe"
WORKER = r"G:\pureskill\csds-cs2\scripts\swarm2\verify_E2_3_indep_bench.py"
ORDER = ["CPU", "GPU", "GPU", "CPU", "CPU", "GPU", "GPU", "CPU"]
ITERS = "100,100,5"


def run_child(device):
    t = time.perf_counter()
    proc = subprocess.run([PY, "-I", "-X", "utf8", WORKER, device, ITERS],
                          capture_output=True, text=True, timeout=1500)
    wall = time.perf_counter() - t
    fits, meta, warm = [], None, None
    for line in proc.stdout.splitlines():
        if line.startswith("INDEP_FIT "):
            fits.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("INDEP_META "):
            meta = json.loads(line.split(" ", 1)[1])
        elif line.startswith("INDEP_WARMUP "):
            warm = json.loads(line.split(" ", 1)[1])
    return {"device": device, "exit": proc.returncode, "wall_s": round(wall, 1), "fits": fits,
            "meta": meta, "warm": warm,
            "stderr_tail": proc.stderr[-600:] if proc.returncode else ""}


def exact_perm(gpu, cpu):
    """All splits of the pooled fit times into two groups of the original sizes.
    Returns observed mean(GPU)-mean(CPU), one-sided p (GPU slower), two-sided p, number of splits."""
    pooled = list(gpu) + list(cpu)
    ng = len(gpu)
    obs = statistics.mean(gpu) - statistics.mean(cpu)
    total = sum(pooled)
    n = len(pooled)
    diffs = []
    for idx in itertools.combinations(range(n), ng):
        sg = sum(pooled[i] for i in idx)
        mg = sg / ng
        mc = (total - sg) / (n - ng)
        diffs.append(mg - mc)
    m = len(diffs)
    one = sum(1 for d in diffs if d >= obs - 1e-9) / m
    two = sum(1 for d in diffs if abs(d) >= abs(obs) - 1e-9) / m
    return obs, one, two, m


def main():
    print(f"VERIFY_E2_3_INDEP start {time.strftime('%Y-%m-%d %H:%M:%S')} order={ORDER} iters={ITERS}", flush=True)
    runs = []
    for slot, dev in enumerate(ORDER):
        r = run_child(dev)
        r["slot"] = slot
        runs.append(r)
        brief = [(f["iters"], f["fit_seconds"], f["system_cpu_busy_pct"]) for f in r["fits"]]
        print(f"child slot={slot} device={dev} exit={r['exit']} wall_s={r['wall_s']} "
              f"(iters, fit_s, sys_busy_pct)={brief}", flush=True)
        if r["exit"]:
            print("child stderr tail: " + r["stderr_tail"], flush=True)

    rows = []
    for r in runs:
        for f in r["fits"]:
            rows.append({"slot": r["slot"], "device": r["device"], **f})
    print("\nPER-FIT TABLE", flush=True)
    print(f"{'slot':>4} {'dev':>4} {'iters':>5} {'fit_s':>8} {'sys_busy%':>9} {'gpu_util%':>9} "
          f"{'gpu_mem_pre':>11} {'gpu_mem_max':>11} {'trees':>5} {'task':>5} error", flush=True)
    for x in rows:
        print(f"{x['slot']:>4} {x['device']:>4} {x['iters']:>5} {x['fit_seconds']:>8.3f} "
              f"{str(x['system_cpu_busy_pct']):>9} {str(x['gpu_util_mean_pct']):>9} "
              f"{str(x['gpu_mem_pre_min_mib']):>11} {str(x['gpu_mem_max_mib']):>11} "
              f"{str(x['n_trees']):>5} {str(x['task_type_effective']):>5} {x['error']}", flush=True)

    print("\nDATA CHECK", flush=True)
    for r in runs:
        m = r["meta"] or {}
        print(f"slot {r['slot']} {r['device']}: n_train={m.get('n_train')} n_feat={m.get('n_feat')} "
              f"train_positive_rate={m.get('train_positive_rate')} catboost={m.get('catboost')} "
              f"warmup={(r['warm'] or {}).get('warmup_seconds')}", flush=True)

    big = [x for x in rows if x["iters"] == 100 and not x["error"]]
    gpu100 = [x["fit_seconds"] for x in big if x["device"] == "GPU"]
    cpu100 = [x["fit_seconds"] for x in big if x["device"] == "CPU"]
    gpu_busy = [x["system_cpu_busy_pct"] for x in big if x["device"] == "GPU"]
    cpu_busy = [x["system_cpu_busy_pct"] for x in big if x["device"] == "CPU"]
    print("\nCLAIM-SIZE SUMMARY (100 trees, 100000 x 50)", flush=True)
    for name, xs, busy in (("GPU", gpu100, gpu_busy), ("CPU1", cpu100, cpu_busy)):
        if xs:
            cv = 100 * statistics.pstdev(xs) / statistics.mean(xs)
            print(f"{name}: n={len(xs)} fits={xs} median={statistics.median(xs):.3f} "
                  f"mean={statistics.mean(xs):.3f} min={min(xs):.3f} max={max(xs):.3f} cv={cv:.1f}% "
                  f"sys_busy={busy}", flush=True)
    if gpu100 and cpu100:
        print(f"ratio of medians GPU/CPU1 = {statistics.median(gpu100) / statistics.median(cpu100):.4f}", flush=True)
        obs, one, two, m = exact_perm(gpu100, cpu100)
        print(f"exact permutation over {m} splits: observed mean(GPU)-mean(CPU) = {obs:.3f} s; "
              f"one-sided p (GPU slower) = {one:.4f}; two-sided p = {two:.4f}", flush=True)

    print("\nBLOCK RATIOS (adjacent child pairs, ABBA)", flush=True)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    by_slot = {r["slot"]: r for r in runs}
    block_ratios = []
    for a, b in pairs:
        ra, rb = by_slot[a], by_slot[b]
        cpu_r, gpu_r = (ra, rb) if ra["device"] == "CPU" else (rb, ra)
        c = [f["fit_seconds"] for f in cpu_r["fits"] if f["iters"] == 100 and not f["error"]]
        g = [f["fit_seconds"] for f in gpu_r["fits"] if f["iters"] == 100 and not f["error"]]
        if c and g:
            ratio = statistics.mean(g) / statistics.mean(c)
            block_ratios.append(ratio)
            cb_busy = [f["system_cpu_busy_pct"] for f in cpu_r["fits"] if f["iters"] == 100]
            gb_busy = [f["system_cpu_busy_pct"] for f in gpu_r["fits"] if f["iters"] == 100]
            print(f"slots {a}-{b}: CPU1 mean={statistics.mean(c):.3f} GPU mean={statistics.mean(g):.3f} "
                  f"ratio GPU/CPU1={ratio:.4f} CPU busy={cb_busy} GPU busy={gb_busy}", flush=True)
    if block_ratios:
        print(f"median block ratio GPU/CPU1 = {statistics.median(block_ratios):.4f}; "
              f"blocks with GPU slower = {sum(1 for x in block_ratios if x > 1)} of {len(block_ratios)}", flush=True)

    print("\nOVERHEAD PROBE (ASSUMPTION: fit time linear in trees; intercept = 5-tree fit)", flush=True)
    t5 = {"GPU": [], "CPU": []}
    for x in rows:
        if x["iters"] == 5 and not x["error"]:
            t5[x["device"]].append(x["fit_seconds"])
    for dev in ("GPU", "CPU"):
        if t5[dev] and gpu100 and cpu100:
            med100 = statistics.median(gpu100 if dev == "GPU" else cpu100)
            m5 = statistics.median(t5[dev])
            slope = (med100 - m5) / 95.0
            print(f"{dev}: 5-tree fits={t5[dev]} median5={m5:.3f} s; median100={med100:.3f} s; "
                  f"slope={slope * 1000:.1f} ms/tree; intercept~{m5:.3f} s", flush=True)

    print("\nTREE COUNT CHECK", flush=True)
    for x in rows:
        if x.get("n_trees") is not None and x["n_trees"] != x["iters"]:
            print(f"MISMATCH slot {x['slot']} requested {x['iters']} got {x['n_trees']}", flush=True)
    print("tree counts checked", flush=True)
    print(f"VERIFY_E2_3_INDEP_DONE {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
