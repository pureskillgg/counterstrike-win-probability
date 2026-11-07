"""verify_E2_3 phase 3: claim-size paired check with the full CPU thread-count sweep.

The first run (verify_E2_3_bench.py) measured CPU at thread_count=1 only at the claim size (100k rows,
50 features, 100 trees). This run adds thread_count 4 and 16 at the same size, 3 rounds, with the
order of GPU, CPU1, CPU4, CPU16 rotated each round. Helpers come from verify_E2_3_bench.py (imported;
its main() is not run). ASSUMPTION: same synthetic data, hyperparameters and gpu_ram_part 0.2 as before.

Run with the cs2gpu_alt interpreter (see verify_E2_3_bench.py for the path). Output is appended to
logs\swarm2\verify_E2_3.log by the caller's >> redirect. Writes nothing else.
"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_E2_3_bench as base  # noqa: E402

P3 = dict(n_train=100000, n_valid=50000, n_feat=50, iters=100)
ROUNDS = 3


def main():
    print("PHASE3_META " + json.dumps({"claim_size": P3, "rounds": ROUNDS, "devices": ["GPU", "CPU1", "CPU4", "CPU16"],
                                       "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "sampler_interval_s": 1.0}), flush=True)
    results = []
    sampler = base.GpuSampler()
    sampler.start()
    time.sleep(2.0)
    try:
        Xtr, ytr = base.make_data(P3["n_train"], P3["n_valid"], P3["n_feat"])
        base.gpu_warmup(Xtr, ytr)
        base_order = ["GPU", "CPU1", "CPU4", "CPU16"]
        for rnd in range(ROUNDS):
            order = base_order[rnd:] + base_order[:rnd]
            for dev in order:
                results.append(base.timed_fit(dev, P3["iters"], Xtr, ytr, "phase3_claim_size_threads", rnd, sampler))
    finally:
        sampler.stop()

    rs = [r for r in results if r["error"] is None]
    devs = ["GPU", "CPU1", "CPU4", "CPU16"]
    per = {}
    for d in devs:
        ts = [r["fit_seconds"] for r in rs if r["device"] == d]
        per[d] = {"fits": ts, "median_fit_s": round(statistics.median(ts), 3) if ts else None,
                  "min_fit_s": min(ts) if ts else None, "max_fit_s": max(ts) if ts else None,
                  "system_cpu_busy_pct": [r["system_cpu_busy_pct"] for r in rs if r["device"] == d],
                  "gpu_util_mean_pct": [r.get("gpu_util_mean_pct") for r in rs if r["device"] == d]}
    by_key = {(r["round"], r["device"]): r["fit_seconds"] for r in rs}
    ratios = {}
    for d in devs[1:]:
        ratios[f"GPU_over_{d}_per_round"] = [round(by_key[(k, "GPU")] / by_key[(k, d)], 3)
                                             for k in range(ROUNDS) if (k, "GPU") in by_key and (k, d) in by_key]
    best = min(devs[1:], key=lambda d: per[d]["median_fit_s"] if per[d]["median_fit_s"] is not None else 1e9)
    summary = {"per_device": per, "ratios": ratios, "best_cpu_by_median": best,
               "median_ratio_GPU_over_best_cpu": round(per["GPU"]["median_fit_s"] / per[best]["median_fit_s"], 3)}
    print("PHASE3_SUMMARY " + json.dumps(summary, indent=1), flush=True)
    print("PHASE3_DONE " + time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
