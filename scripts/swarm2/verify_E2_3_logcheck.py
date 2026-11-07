"""VERIFY E2-3, step 1: recompute the claimed CatBoost GPU-vs-CPU numbers from the raw E2 logs.

Reads only the existing E2 logs (no CatBoost import, no GPU use) and prints:
  - every E2_BENCH_JSON record with its config and load fields (or 'absent')
  - the claimed medians and ratio, recomputed from the per-rep fit times
  - an exact permutation test on the 2-vs-2 claim (all 6 splits)
  - the 20k-row GPU spread and the 16-thread CPU smoke run (per unit of work)

Usage (any interpreter with the standard library):
  "C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe" -I -X utf8 verify_E2_3_logcheck.py
"""
import itertools
import json
import os
import statistics

LOG_DIR = r"G:\pureskill\csds-cs2\logs\swarm2"
FILES = [
    "E2_bench_cb_CPU.log",      # 100k x 50, 100 trees, CPU thread_count=1 (claim CPU arm)
    "E2_retry_cb_GPU.log",      # 100k x 50, 100 trees, GPU gpu_ram_part=0.2 (claim GPU arm)
    "E2_bench_cb_GPU.log",      # first GPU attempt: CUDA OOM at warm-up, no timing
    "E2_smoke_cb_CPU.log",      # 20k x 20, 20 trees, CPU thread_count=16
    "E2_smoke_cb_GPU.log",      # 20k x 20, 20 trees, GPU (no load fields in this record)
    "E2_smoke3_cb_GPU.log",     # same config as smoke_cb_GPU, later, load fields present
    "E2_smoke2_cb_GPU.log",     # empty file
]
PREFIX = "E2_BENCH_JSON "


def load_records():
    out = []
    for name in FILES:
        path = os.path.join(LOG_DIR, name)
        size = os.path.getsize(path)
        text = open(path, encoding="utf-8", errors="replace").read()
        rec = None
        for line in text.splitlines():
            if line.startswith(PREFIX):
                rec = json.loads(line[len(PREFIX):])
        out.append((name, size, rec, text))
    return out


def exact_perm_test(a, b):
    """Exact two-sample permutation test on the difference of means (b minus a).
    Returns (observed diff, one-sided p for b > a, two-sided p)."""
    pooled = list(a) + list(b)
    na = len(a)
    obs = statistics.mean(b) - statistics.mean(a)
    diffs = []
    for idx in itertools.combinations(range(len(pooled)), na):
        sel = set(idx)
        aa = [pooled[i] for i in range(len(pooled)) if i in sel]
        bb = [pooled[i] for i in range(len(pooled)) if i not in sel]
        diffs.append(statistics.mean(bb) - statistics.mean(aa))
    n = len(diffs)
    one = sum(1 for d in diffs if d >= obs - 1e-12) / n
    two = sum(1 for d in diffs if abs(d) >= abs(obs) - 1e-12) / n
    return obs, one, two, n


def main():
    recs = load_records()
    print("== E2 logs: records and load fields ==")
    for name, size, rec, text in recs:
        print(f"\n[{name}] bytes={size}")
        if rec is None:
            first = text.strip().splitlines()[:1]
            print(f"  no E2_BENCH_JSON record; first line: {first[0] if first else '(empty)'}")
            continue
        print(f"  device={rec['device']} n_train={rec['n_train']} n_feat={rec['n_feat']} "
              f"iters={rec['iters']} repeats={rec['repeats']} status={rec['status']}")
        print(f"  params.task_type={rec['params'].get('task_type')} "
              f"thread_count={rec['params'].get('thread_count')} gpu_ram_part={rec.get('gpu_ram_part')} "
              f"gpu_warmup_seconds={rec.get('gpu_warmup_seconds')}")
        for r in rec.get("reps", []):
            print(f"  rep {r['rep']}: fit_seconds={r['fit_seconds']} valid_logloss={r['valid_logloss']} "
                  f"system_cpu_busy_mean_pct={r.get('system_cpu_busy_mean_pct', 'absent')} "
                  f"gpu_util_mean_pct={r.get('gpu_util_mean_pct', 'absent')} "
                  f"gpu_mem_baseline_mib={r.get('gpu_mem_baseline_mib', 'absent')}")
        print(f"  median_fit_seconds (json)={rec.get('median_fit_seconds')}")

    by_name = {name: rec for name, _, rec, _ in recs}

    print("\n== Recompute the claimed medians and ratio ==")
    cpu_fit = [r["fit_seconds"] for r in by_name["E2_bench_cb_CPU.log"]["reps"]]
    gpu_fit = [r["fit_seconds"] for r in by_name["E2_retry_cb_GPU.log"]["reps"]]
    cpu_med = statistics.median(cpu_fit)
    gpu_med = statistics.median(gpu_fit)
    print(f"CPU (thread_count=1, 100k) reps={cpu_fit} median={cpu_med:.3f} (claim 20.241)")
    print(f"GPU (100k) reps={gpu_fit} median={gpu_med:.3f} (claim 31.788)")
    print(f"ratio GPU/CPU median = {gpu_med / cpu_med:.4f} (claim-implied 31.788/20.241 = 1.5703)")
    print(f"same-config GPU rep spread (max/min) = {max(gpu_fit) / min(gpu_fit):.3f}")
    print(f"CPU rep spread (max/min) = {max(cpu_fit) / min(cpu_fit):.3f}")
    print(f"median of 2 reps equals mean of 2 reps: CPU {cpu_med:.3f} vs {statistics.mean(cpu_fit):.3f}, "
          f"GPU {gpu_med:.3f} vs {statistics.mean(gpu_fit):.3f}")

    obs, one, two, n = exact_perm_test(cpu_fit, gpu_fit)
    print(f"\nExact permutation test on mean(fit): observed GPU-CPU diff = {obs:.3f} s, "
          f"splits enumerated = {n}")
    print(f"  one-sided p (GPU slower) = {one:.4f}   two-sided p = {two:.4f}")
    print("  note: with 2 vs 2 samples and perfect separation, the smallest achievable one-sided p is 1/6.")

    print("\n== Load-field check for the 20k-row GPU claim (12.465 s vs 63.642 s) ==")
    s_fast = by_name["E2_smoke_cb_GPU.log"]["reps"][0]
    s_slow = by_name["E2_smoke3_cb_GPU.log"]["reps"][0]
    print(f"fast run fit={s_fast['fit_seconds']} s, warmup={by_name['E2_smoke_cb_GPU.log']['gpu_warmup_seconds']} s, "
          f"gpu_mem_baseline={s_fast.get('gpu_mem_baseline_mib')} MiB, "
          f"system_cpu_busy_mean_pct={s_fast.get('system_cpu_busy_mean_pct', 'ABSENT')}")
    print(f"slow run fit={s_slow['fit_seconds']} s, warmup={by_name['E2_smoke3_cb_GPU.log']['gpu_warmup_seconds']} s, "
          f"gpu_mem_baseline={s_slow.get('gpu_mem_baseline_mib')} MiB, "
          f"system_cpu_busy_mean_pct={s_slow.get('system_cpu_busy_mean_pct', 'ABSENT')}")
    print(f"slow/fast fit ratio = {s_slow['fit_seconds'] / s_fast['fit_seconds']:.3f}")
    print("  the fast run has no CPU-busy field, so 'under full load' is only shown for the slow run.")

    print("\n== 16-thread CPU smoke run vs 1-thread CPU bench (crude work scaling) ==")
    cpu16 = by_name["E2_smoke_cb_CPU.log"]
    w16 = cpu16["n_train"] * cpu16["n_feat"] * cpu16["iters"]
    w1 = 100000 * 50 * 100
    t16 = cpu16["reps"][0]["fit_seconds"]
    t1 = cpu_med
    print(f"work units (n_train*n_feat*iters, ASSUMPTION fit time ~ linear in work): "
          f"16-thread smoke={w16:.3g}, 1-thread bench={w1:.3g}, ratio={w1 / w16:.1f}x")
    print(f"fit time: 16-thread smoke={t16} s, 1-thread bench median={t1:.3f} s")
    print(f"implied seconds per unit of work, 16-thread / 1-thread = "
          f"{(t16 / w16) / (t1 / w1):.1f}x (ASSUMPTION: linear scaling, fixed overhead ignored)")
    print("  reading: the 16-thread CPU run was far slower per unit of work than the 1-thread run,")
    print("  so the CPU baseline itself was contention-bound. The 1-thread CPU arm is not the best CPU config.")


if __name__ == "__main__":
    main()
