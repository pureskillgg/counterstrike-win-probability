"""verify_E1_3, part 3: put E1's logged per-tree times next to this session's live CPU per-tree times.

Stdlib only. Reads logs/swarm2/E1_bench.log (E1 numbers) and the PART 2 section of logs/swarm2/verify_E1_3.log
(live numbers from verify_E1_3_rerun.py). Prints computed ratios; the caller appends stdout to the log.
Cross-session and cross-environment ratios are ASSUMPTION-level: E1's CUDA run and E1's CPU runs came from the C:
cs2gpu env, the live CPU runs from the G: project env.
"""

import json
import re
import statistics
from pathlib import Path

ROOT = Path(r"G:\pureskill\csds-cs2")
BENCH = ROOT / "logs" / "swarm2" / "E1_bench.log"
VERIFY = ROOT / "logs" / "swarm2" / "verify_E1_3.log"
E1_TREES = 200
LIVE_TREES = 20


def out(*args):
    print(*args, flush=True)


def main():
    bench = BENCH.read_text(encoding="utf-8")
    m = re.search(r"SUMMARY_JSON_BEGIN\n(.*?)\nSUMMARY_JSON_END", bench, re.S)
    res = json.loads(m.group(1))["results"]
    e1_def = res["cpu_default"]["train_times_s"]
    e1_t1 = res["cpu_1thread"]["train_times_s"]
    e1_cu = res["cuda"]["train_times_s"]
    e1_def_pt = [t / E1_TREES for t in e1_def]
    e1_t1_pt = [t / E1_TREES for t in e1_t1]
    e1_cu_pt = [t / E1_TREES for t in e1_cu]

    ver = VERIFY.read_text(encoding="utf-8")
    part2 = ver.split("== PART 2", 1)[1]
    live = {"t1": [], "default": []}
    for mm in re.finditer(r"^\s+(t1|default)\s+rep\d: ([\d.]+) s\s+\(([\d.]+) s/tree\)", part2, re.M):
        live[mm.group(1)].append(float(mm.group(2)) / LIVE_TREES)
    small = {}
    for mm in re.finditer(r"^\s+nthread=\s*(\d+) rep\d: ([\d.]+) s", part2, re.M):
        small.setdefault(int(mm.group(1)), []).append(float(mm.group(2)))

    out("== PART 3: E1 per-tree times vs this session's live CPU per-tree times ==")
    out(f"E1 (C: cs2gpu env, 200 trees, 1M x 50), s/tree: default runs={[round(x, 4) for x in e1_def_pt]} "
        f"1thread runs={[round(x, 4) for x in e1_t1_pt]} cuda runs={[round(x, 4) for x in e1_cu_pt]}")
    out(f"live (G: project env, {LIVE_TREES} trees, 1M x 50), s/tree: default runs={[round(x, 4) for x in live['default']]} "
        f"1thread runs={[round(x, 4) for x in live['t1']]}")
    out("")

    ld = statistics.median(live["default"])
    lt = statistics.median(live["t1"])
    ed = statistics.median(e1_def_pt)
    et = statistics.median(e1_t1_pt)
    ec = statistics.median(e1_cu_pt)
    out("-- same CPU config, E1 vs live (s/tree, medians) --")
    out(f"default threads: E1 {ed:.4f} (n=1)  live {ld:.4f} (n={len(live['default'])})  "
        f"live/E1 = {ld / ed:.4f}  (E1 is {ed / ld:.1f}x slower per tree)")
    out(f"1 thread:        E1 {et:.4f} (n=2)  live {lt:.4f} (n={len(live['t1'])})  "
        f"live/E1 = {lt / et:.4f}  (E1 is {et / lt:.2f}x slower per tree)")
    out("")

    out("-- speedup of E1's CUDA median (C: env) over each CPU baseline --")
    out(f"E1 CUDA median per tree = {ec:.4f} s/tree")
    out(f"E1 internal:   default/CUDA = {ed / ec:.2f}x   1thread/CUDA = {et / ec:.2f}x")
    out(f"cross-env, ASSUMPTION (live CPU per tree / E1 CUDA per tree): "
        f"default/CUDA = {ld / ec:.2f}x (per-rep range {min(live['default']) / ec:.2f}-{max(live['default']) / ec:.2f}x)  "
        f"1thread/CUDA = {lt / ec:.2f}x (per-rep range {min(live['t1']) / ec:.2f}-{max(live['t1']) / ec:.2f}x)")
    out(f"E1 CUDA spread alone (min..max per tree): {min(e1_cu_pt):.4f}..{max(e1_cu_pt):.4f} s/tree")
    out("")

    out("-- within-session stability (live) --")
    pairs = [d / t for d, t in zip(live["default"], live["t1"])]
    out(f"live default/1thread per rep: {[round(p, 3) for p in pairs]}  median {statistics.median(pairs):.3f}")
    out(f"live 1thread spread max/min = {max(live['t1']) / min(live['t1']):.3f}; "
        f"live default spread max/min = {max(live['default']) / min(live['default']):.3f}")
    out("")

    out("-- 100k x 10-tree thread check, live vs E1_threads --")
    for nth in sorted(small):
        v = small[nth]
        out(f"live nthread={nth:2d} runs={v} median={statistics.median(v):.3f}s "
            f"median/nthread1={statistics.median(v) / statistics.median(small[1]):.2f}x")
    out("E1_threads (same shape, C: env): nthread=2 14.304 s (28.49x), nthread=16 11.606 s (23.12x, runs 22.472 and 0.739)")
    out("")
    out("PART 3 DONE")


if __name__ == "__main__":
    main()
