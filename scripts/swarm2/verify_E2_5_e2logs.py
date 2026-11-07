r"""verify_E2_5 recompute from the ORIGINAL E2 raw logs (no new fits).

Reads the E2_BENCH_JSON lines in logs\swarm2\E2_bench_lgb_cpu.log and logs\swarm2\E2_bench_cb_CPU.log,
recomputes the claimed medians, and recomputes the same ratio with the LightGBM dataset build added to
its fit time (CatBoost's fit already includes its own quantization, per notes\swarm2\E2.md section 4).

  <cs2gpu_alt>\\envs\\default\\python.exe -B -I -X utf8 scripts\\swarm2\\verify_E2_5_e2logs.py

Writes nothing except stdout.
"""
import json
import statistics

LGB_LOG = r"G:\pureskill\csds-cs2\logs\swarm2\E2_bench_lgb_cpu.log"
CB_LOG = r"G:\pureskill\csds-cs2\logs\swarm2\E2_bench_cb_CPU.log"


def load(path):
    out = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("E2_BENCH_JSON "):
                out = json.loads(line[len("E2_BENCH_JSON "):])
    if out is None:
        raise SystemExit(f"no E2_BENCH_JSON line in {path}")
    return out


lgb = load(LGB_LOG)
cb = load(CB_LOG)
print(f"LGB record: status={lgb.get('status')} engine={lgb.get('engine')} device={lgb.get('device')} "
      f"n_train={lgb.get('n_train')} iters={lgb.get('iters')} repeats={lgb.get('repeats')} "
      f"num_threads={lgb.get('params', {}).get('num_threads')}")
print(f"CB record:  status={cb.get('status')} engine={cb.get('engine')} device={cb.get('device')} "
      f"n_train={cb.get('n_train')} iters={cb.get('iters')} repeats={cb.get('repeats')} "
      f"thread_count={cb.get('params', {}).get('thread_count')}")

lr = lgb["reps"]
cr = cb["reps"]
for r in lr:
    print(f"LGB rep {r['rep']}: build {r['build_seconds']} fit {r['fit_seconds']} logloss {r['valid_logloss']} "
          f"busy_fit {r.get('system_cpu_busy_mean_pct')}")
for r in cr:
    print(f"CB  rep {r['rep']}: fit {r['fit_seconds']} logloss {r['valid_logloss']} "
          f"busy_fit {r.get('system_cpu_busy_mean_pct')}")

lgb_fit = [r["fit_seconds"] for r in lr]
cb_fit = [r["fit_seconds"] for r in cr]
lgb_build = [r["build_seconds"] for r in lr]
lgb_total = [r["build_seconds"] + r["fit_seconds"] for r in lr]

print("\n-- claim arithmetic --")
print(f"LGB median fit = {statistics.median(lgb_fit):.4f} (claim 10.8495)")
print(f"CB  median fit = {statistics.median(cb_fit):.4f} (claim 20.241)")
print(f"claim-style ratio CB/LGB = {statistics.median(cb_fit) / statistics.median(lgb_fit):.4f} (E2.md: about 2x; 1.865 from its medians)")

print("\n-- scope-matched (LGB build + fit vs CB fit, which includes CB quantization) --")
print(f"LGB build seconds per rep: {lgb_build}, median {statistics.median(lgb_build):.4f}")
print(f"LGB build+fit per rep: {[round(x, 3) for x in lgb_total]}, median {statistics.median(lgb_total):.4f}")
print(f"scope-matched ratio CB fit / LGB (build+fit) = {statistics.median(cb_fit) / statistics.median(lgb_total):.4f}")
paired = [c / t for c, t in zip(cb_fit, lgb_total)]
print(f"paired per-rep ratios CB fit / LGB (build+fit): {[round(x, 3) for x in paired]}")

print("\n-- spread of the claim-style ratio across the two reps --")
pr = [c / f for c, f in zip(cb_fit, lgb_fit)]
print(f"paired per-rep ratios CB fit / LGB fit: {[round(x, 3) for x in pr]} min {min(pr):.3f} max {max(pr):.3f}")
print("E2 machine state: E2.md section 1 reports CPU LoadPercentage at 100 in several samples during the runs.")
