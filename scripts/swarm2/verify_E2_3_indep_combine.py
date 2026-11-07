"""VERIFY E2-3, step 3: pool every claim-size fit we have (claim logs, parallel verifier, this verifier)
and ask how the GPU/CPU(1 thread) fit-time ratio depends on whole-system CPU busy %.

Sources (100000 x 50, 100 trees, same settings):
  claim   : E2_bench_cb_CPU.log (CPU, thread_count=1) and E2_retry_cb_GPU.log (GPU), busy ~100%
  parallel: verify_E2_3.log 'FIT {...}' lines with phase phase1_claim_size (other verifier), busy ~30-48%
  indep   : verify_E2_3.log 'PER-FIT TABLE' rows with iters 100 (this verifier's driver)

Model (ASSUMPTION): ln(fit_seconds) = a_dev + b_dev * busy_pct, separately per device, pooled over sources.
Predicted GPU/CPU1 ratio at busy B = exp((a_GPU - a_CPU) + (b_GPU - b_CPU) * B).
The crossover busy level is where that ratio equals 1. Extrapolation outside the observed busy range is NOT
supported by data; it is shown only to test the claim's load story.

Usage (cs2gpu_alt interpreter):
  "C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe" -I -X utf8 verify_E2_3_indep_combine.py
Reads only. Prints to stdout (the caller appends to logs\\swarm2\\verify_E2_3.log).
"""
import json
import math
import statistics

LOG_DIR = r"G:\pureskill\csds-cs2\logs\swarm2"
VERIFY_LOG = LOG_DIR + r"\verify_E2_3.log"


def claim_rows():
    out = []
    for name, dev in (("E2_bench_cb_CPU.log", "CPU"), ("E2_retry_cb_GPU.log", "GPU")):
        for line in open(LOG_DIR + "\\" + name, encoding="utf-8", errors="replace"):
            if line.startswith("E2_BENCH_JSON "):
                rec = json.loads(line[len("E2_BENCH_JSON "):])
                for r in rec["reps"]:
                    out.append(("claim", dev, r["fit_seconds"], r["system_cpu_busy_mean_pct"]))
    return out


def parallel_and_indep_rows():
    parallel, indep = [], []
    in_table = False
    for line in open(VERIFY_LOG, encoding="utf-8", errors="replace"):
        s = line.rstrip("\n")
        if s.startswith("FIT {"):
            rec = json.loads(s[4:])
            if rec.get("phase") == "phase1_claim_size" and rec.get("iters") == 100 and rec.get("error") is None:
                dev = "GPU" if rec["device"] == "GPU" else "CPU"
                parallel.append(("parallel", dev, rec["fit_seconds"], rec["system_cpu_busy_pct"]))
        if s.strip() == "PER-FIT TABLE":
            in_table = True
            continue
        if in_table:
            if s.strip() == "" or s.startswith("DATA CHECK"):
                in_table = False
                continue
            parts = s.split()
            if len(parts) >= 10 and parts[0].isdigit() and parts[2] == "100":
                if parts[10:] and parts[10] not in ("None", ""):
                    continue
                busy = None if parts[4] == "None" else float(parts[4])
                indep.append(("indep", "GPU" if parts[1] == "GPU" else "CPU", float(parts[3]), busy))
    return parallel, indep


def ols(x, y):
    n = len(x)
    mx, my = statistics.mean(x), statistics.mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return None
    b = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / sxx
    a = my - b * mx
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    return a, b, math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))


def main():
    claim = claim_rows()
    parallel, indep = parallel_and_indep_rows()
    rows = claim + parallel + indep
    print("POOLED CLAIM-SIZE FITS (100000 x 50, 100 trees)")
    print(f"{'source':>9} {'dev':>4} {'fit_s':>8} {'busy_pct':>9}")
    for src, dev, t, busy in sorted(rows, key=lambda r: (r[3] if r[3] is not None else -1)):
        print(f"{src:>9} {dev:>4} {t:>8.3f} {str(busy):>9}")
    print(f"counts: claim={len(claim)} parallel={len(parallel)} indep={len(indep)}")

    usable = [r for r in rows if r[3] is not None]
    fits = {}
    for dev in ("GPU", "CPU"):
        pts = [(r[3], math.log(r[2])) for r in usable if r[1] == dev]
        fits[dev] = ols([p[0] for p in pts], [p[1] for p in pts])
        xs = [p[0] for p in pts]
        print(f"{dev}: n={len(pts)} busy range {min(xs):.1f}-{max(xs):.1f}; "
              f"ln(fit) = {fits[dev][0]:.4f} + {fits[dev][1]:.5f} * busy (resid sd {fits[dev][2]:.4f})")

    aG, bG, _ = fits["GPU"]
    aC, bC, _ = fits["CPU"]
    print("\nPREDICTED GPU/CPU1 RATIO (model, ASSUMPTION log-linear in busy%)")
    for B in (30, 40, 50, 60, 80, 100):
        r = math.exp((aG - aC) + (bG - bC) * B)
        print(f"  busy {B:>3}%: ratio GPU/CPU1 = {r:.3f}")
    if bG != bC:
        cross = (aC - aG) / (bG - bC)
        print(f"crossover busy % where ratio = 1: {cross:.1f} (ASSUMPTION model; check against observed range)")
    print(f"slope of ln(fit) per busy %: GPU={bG:.5f}, CPU1={bC:.5f} "
          f"(GPU more load-sensitive if GPU > CPU1)")

    print("\nDIRECT COMPARISON BY SOURCE (median fit, GPU/CPU1)")
    for src in ("claim", "parallel", "indep"):
        g = [r[2] for r in rows if r[0] == src and r[1] == "GPU"]
        c = [r[2] for r in rows if r[0] == src and r[1] == "CPU"]
        if g and c:
            bg = [r[3] for r in rows if r[0] == src and r[1] == "GPU" and r[3] is not None]
            bc = [r[3] for r in rows if r[0] == src and r[1] == "CPU" and r[3] is not None]
            print(f"{src:>9}: n_GPU={len(g)} n_CPU={len(c)} median GPU={statistics.median(g):.3f} "
                  f"median CPU1={statistics.median(c):.3f} ratio={statistics.median(g) / statistics.median(c):.3f} "
                  f"busy GPU mean={statistics.mean(bg) if bg else None} CPU mean={statistics.mean(bc) if bc else None}")


if __name__ == "__main__":
    main()
