"""verify_E1_3, part 1: recompute the E1-3 speedups from the raw logged times and check the contention evidence.

Stdlib only. Reads logs/swarm2/E1_bench.log and logs/swarm2/E1_threads.log (written by E1; not modified here).
No timing happens in this script. Output goes to stdout; the caller redirects it into logs/swarm2/verify_E1_3.log.
"""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(r"G:\pureskill\csds-cs2")
BENCH = ROOT / "logs" / "swarm2" / "E1_bench.log"
THREADS = ROOT / "logs" / "swarm2" / "E1_threads.log"
N_TREES = 200  # from the E1 workload (ASSUMPTION carried over from E1)


def out(*args):
    print(*args, flush=True)


def main():
    bench_txt = BENCH.read_text(encoding="utf-8")
    m = re.search(r"SUMMARY_JSON_BEGIN\n(.*?)\nSUMMARY_JSON_END", bench_txt, re.S)
    summ = json.loads(m.group(1))
    res = summ["results"]

    out("== PART 1: recompute E1-3 from logged raw times (no new timing) ==")
    out(f"source: {BENCH}")
    out(f"workload (as logged): {summ['workload']}")
    out("")

    out("-- per-config raw runs (as logged) --")
    med = {}
    for name in ("cpu_default", "cpu_1thread", "cuda"):
        t = res[name]["train_times_s"]
        med[name] = statistics.median(t)
        sd = statistics.stdev(t) if len(t) > 1 else float("nan")
        out(f"{name:12s} n={len(t)} runs={t} median={med[name]:.3f} (logged {res[name]['train_median_s']}) "
            f"s/tree={med[name] / N_TREES:.4f} (logged {res[name]['s_per_tree_median']}) "
            f"min={min(t):.3f} max={max(t):.3f} max/min={max(t) / min(t):.3f} "
            f"mean={statistics.mean(t):.3f} stdev={sd:.3f} cv={sd / statistics.mean(t) if len(t) > 1 else float('nan'):.3f}")
    out("")

    out("-- speedups recomputed from medians --")
    d, o, c = med["cpu_default"], med["cpu_1thread"], med["cuda"]
    out(f"cuda vs cpu_default = {d:.3f} / {c:.3f} = {d / c:.3f}   (logged {summ['speedup_cuda_vs_cpu_default_median']})")
    out(f"cuda vs cpu_1thread = {o:.3f} / {c:.3f} = {o / c:.3f}   (logged {summ['speedup_cuda_vs_cpu_1thread_median']})")
    t1 = res["cpu_1thread"]["train_times_s"]
    cu = res["cuda"]["train_times_s"]
    df = res["cpu_default"]["train_times_s"]
    out(f"cuda vs cpu_1thread, bounded by the logged run spread on both sides: "
        f"{min(t1) / max(cu):.2f}x to {max(t1) / min(cu):.2f}x")
    out(f"cuda vs cpu_default, bounded by CUDA run spread ONLY (default has 1 run, its own noise is unbounded): "
        f"{df[0] / max(cu):.2f}x to {df[0] / min(cu):.2f}x")
    out("")

    out("-- does the CPU default-thread config ever beat 1 thread? (E1_threads sweep, 100k rows, 10 trees) --")
    thr_txt = THREADS.read_text(encoding="utf-8")
    rows = {}
    pat = re.compile(r"device=(cpu|cuda)\s+nthread=(\S+)\s+median_s=\s*([\d.]+)\s+s_per_tree=\s*([\d.]+)\s+runs=\[([^\]]*)\]")
    for mm in pat.finditer(thr_txt):
        key = f"{mm.group(1)}:{mm.group(2)}"
        runs = [float(x) for x in mm.group(5).split(",")]
        rows[key] = {"median": float(mm.group(3)), "runs": runs}
    base = rows["cpu:1"]
    best1 = min(base["runs"])
    any_multi_faster = False
    for key, v in rows.items():
        if key.startswith("cpu:") and key != "cpu:1" and v["median"] < base["median"]:
            any_multi_faster = True
        out(f"{key:12s} median={v['median']:.3f}s runs={v['runs']} "
            f"median/cpu1={v['median'] / base['median']:.2f}x best_run/best_cpu1={min(v['runs']) / best1:.2f}x")
    out(f"any cpu config with nthread>1 faster than nthread=1 by median: {any_multi_faster}")
    best16 = min(rows["cpu:16"]["runs"])
    out(f"cpu nthread=16 (= default on this 16-logical-CPU host) best single run = {best16:.3f}s "
        f"vs cpu nthread=1 best run = {best1:.3f}s  -> ratio {best16 / best1:.2f}x (>1 means slower)")
    out("")

    out("-- contention snapshots in E1_bench.log --")
    snap_re = re.compile(r"^\[([^\]]+)\] gpu\(mem,util\)=(\d+) MiB, (\d+) % \| host_cpu_load%=(\d+)", re.M)
    snaps = snap_re.findall(bench_txt)
    for tag, mem, util, cpu in snaps:
        out(f"  {tag:34s} gpu_mem_MiB={mem:>6s} gpu_util%={util:>3s} host_cpu_load%={cpu}")
    cpu_vals = sorted({int(s[3]) for s in snaps})
    out(f"host_cpu_load% values across {len(snaps)} snapshots: {cpu_vals}")
    cuda_snaps = [s for s in snaps if s[0].startswith("cuda")]
    out(f"GPU util% in snapshots taken around the CUDA timed runs: {[s[2] for s in cuda_snaps]} "
        f"(coarse, nvidia-smi sample; CUDA timed runs were not sampled individually)")
    out("")

    out("-- other checks on the logged JSON --")
    out(f"holdout logloss cpu_default={res['cpu_default']['holdout_logloss']} "
        f"cpu_1thread={res['cpu_1thread']['holdout_logloss']} cuda={res['cuda']['holdout_logloss']}")
    out(f"holdout pred diffs: {summ['holdout_pred_diff']}")
    out(f"config_nthread logged: default={res['cpu_default']['config_nthread']} "
        f"1thread={res['cpu_1thread']['config_nthread']} cuda_device={res['cuda']['config_device']}")
    out(f"warm-up seconds (5 rounds, untimed): default={res['cpu_default']['warmup_5rounds_s']} "
        f"1thread={res['cpu_1thread']['warmup_5rounds_s']} cuda={res['cuda']['warmup_5rounds_s']}")
    out(f"per-tree at 1M rows vs 100k rows, 1 thread: {res['cpu_1thread']['s_per_tree_median']} vs 0.0502 (E1_threads)")
    out("")
    out("PART 1 DONE")


if __name__ == "__main__":
    main()
