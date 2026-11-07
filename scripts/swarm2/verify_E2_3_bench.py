"""verify_E2_3: independent recomputation of the E2-3 claim (notes\\swarm2\\E2.md).

Claim under test: under contention, CatBoost GPU median fit 31.788 s was slower than
CatBoost CPU (thread_count=1) median fit 20.241 s at 100k train rows.

Step 0 recomputes the claimed medians from the raw E2 JSON records (read-only).

Phase 1 (claim size, ASSUMPTION: same synthetic data and hyperparameters as E2_bench.py):
  100000 train rows, 50 features, 100 trees, depth 6, border_count 128, Logloss, seed 0,
  GPU gpu_ram_part 0.2 with one untimed 5-tree warm-up. 3 rounds of (GPU fit, CPU thread_count=1 fit),
  order alternated each round so linear drift over the run cancels in the paired ratio.
Phase 2 (smoke size, ASSUMPTION): 20000 train rows, 20 features, 20 trees.
  3 rounds of GPU, CPU thread_count 16, CPU thread_count 4, CPU thread_count 1, rotated order.
  Tests whether the CPU baseline's thread count changes the verdict.

Timing: time.perf_counter around model.fit only. Whole-system CPU busy % comes from
GetSystemTimes over the same fit window. GPU memory and utilization come from nvidia-smi samples
taken once per second (WDDM hides per-process VRAM, so these are whole-card values).
The sampler thread adds a small amount of load (ASSUMPTION, noted in the log).

Interpreter: the project env (G:\\pureskill\\csds-cs2\\.pixi\\envs\\default) has no catboost, so this
runs with the cs2gpu_alt interpreter that holds catboost 1.2.10 (read-only use, nothing installed):
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu_alt-16300978804608365045\\envs\\default\\python.exe -I -X utf8 <this file>
Writes nothing to disk. The caller redirects stdout to logs\\swarm2\\verify_E2_3.log.
"""
import ctypes
import json
import statistics
import subprocess
import sys
import threading
import time
from ctypes import wintypes

import numpy as np
import catboost as cb

SEED = 20241
GPU_RAM_PART = 0.2  # ASSUMPTION: same value as the E2 benchmark
P1 = dict(n_train=100000, n_valid=50000, n_feat=50, iters=100)
P2 = dict(n_train=20000, n_valid=5000, n_feat=20, iters=20)
P1_ROUNDS = 3
P2_ROUNDS = 3

CLAIM_LOGS = {
    "cpu_thread1": r"G:\pureskill\csds-cs2\logs\swarm2\E2_bench_cb_CPU.log",
    "gpu_retry": r"G:\pureskill\csds-cs2\logs\swarm2\E2_retry_cb_GPU.log",
    "gpu_smoke_fast": r"G:\pureskill\csds-cs2\logs\swarm2\E2_smoke_cb_GPU.log",
    "gpu_smoke_slow": r"G:\pureskill\csds-cs2\logs\swarm2\E2_smoke3_cb_GPU.log",
    "cpu16_smoke": r"G:\pureskill\csds-cs2\logs\swarm2\E2_smoke_cb_CPU.log",
}


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


def _ft(ft):
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def sys_cpu():
    idle, kern, user = FILETIME(), FILETIME(), FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
    # kernel time includes idle time on Windows
    return _ft(idle), _ft(kern) + _ft(user)


def busy_pct(a, b):
    d_idle = b[0] - a[0]
    d_total = b[1] - a[1]
    return round(100.0 * (1.0 - d_idle / d_total), 1) if d_total > 0 else None


class GpuSampler:
    def __init__(self, interval=1.0):
        self.samples = []  # (perf_counter time, memory.used MiB, utilization.gpu %)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _query(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
            mem, util = (float(v) for v in out.split(","))
            return mem, util
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            q = self._query()
            if q is not None:
                self.samples.append((time.perf_counter(), q[0], q[1]))
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=30)

    def window(self, t0, t1):
        inside = [s for s in self.samples if t0 <= s[0] <= t1]
        if not inside:
            return {"gpu_samples": 0, "gpu_mem_max_mib": None,
                    "gpu_util_mean_pct": None, "gpu_util_max_pct": None}
        return {"gpu_samples": len(inside),
                "gpu_mem_max_mib": max(s[1] for s in inside),
                "gpu_util_mean_pct": round(statistics.mean(s[2] for s in inside), 1),
                "gpu_util_max_pct": max(s[2] for s in inside)}


def make_data(n_train, n_valid, n_feat):
    # Same generator as scripts\swarm2\E2_bench.py (seed 20241, Bernoulli of a fixed nonlinear logit).
    rng = np.random.default_rng(SEED)
    n = n_train + n_valid
    X = rng.standard_normal((n, n_feat), dtype=np.float32)
    f = (1.0 * X[:, 0] - 0.8 * X[:, 1] * X[:, 2] + 0.6 * np.sin(2.0 * X[:, 3])
         + 0.5 * (X[:, 4] > 0) * X[:, 5] - 0.3 * X[:, 6] ** 2 + 0.2 * X[:, 7]).astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-f))
    y = (rng.random(n) < p).astype(np.int32)
    return X[:n_train], y[:n_train]


def cb_params(dev, iters):
    p = dict(iterations=iters, depth=6, learning_rate=0.1, loss_function="Logloss",
             border_count=128, random_seed=0, verbose=0, allow_writing_files=False)
    if dev == "GPU":
        p.update(task_type="GPU", devices="0", gpu_ram_part=GPU_RAM_PART)
    else:
        p.update(task_type="CPU", thread_count=int(dev[3:]))
    return p


def timed_fit(dev, iters, Xtr, ytr, phase, rnd, sampler):
    model = cb.CatBoostClassifier(**cb_params(dev, iters))
    c0 = sys_cpu()
    t0 = time.perf_counter()
    err = None
    try:
        model.fit(Xtr, ytr)
    except Exception as exc:  # a native CUDA abort would still kill the process
        err = repr(exc)
    t1 = time.perf_counter()
    c1 = sys_cpu()
    rec = {"phase": phase, "round": rnd, "device": dev, "iters": iters,
           "n_train": int(len(ytr)), "n_feat": int(Xtr.shape[1]),
           "fit_seconds": round(t1 - t0, 3), "system_cpu_busy_pct": busy_pct(c0, c1),
           "task_type_effective": str(model.get_all_params().get("task_type")) if err is None else None,
           "n_trees": int(model.tree_count_) if err is None else None,
           "error": err}
    rec.update(sampler.window(t0, t1))
    print("FIT " + json.dumps(rec), flush=True)
    return rec


def gpu_warmup(Xtr, ytr):
    w = cb.CatBoostClassifier(**cb_params("GPU", 5))
    t0 = time.perf_counter()
    w.fit(Xtr[:20000], ytr[:20000])
    print(f"WARMUP GPU 5 trees 20k rows: {time.perf_counter() - t0:.3f} s", flush=True)


def claim_recompute():
    """Recompute the claimed medians from the E2 JSON records in the raw logs (read-only)."""
    out = {}
    for name, path in CLAIM_LOGS.items():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                line = [ln for ln in fh if ln.startswith("E2_BENCH_JSON ")]
            if not line:
                out[name] = {"error": "no E2_BENCH_JSON line"}
                continue
            rec = json.loads(line[-1][len("E2_BENCH_JSON "):])
            fits = [r["fit_seconds"] for r in rec.get("reps", [])]
            out[name] = {"device": rec["device"], "n_train": rec["n_train"], "n_feat": rec["n_feat"],
                         "iters": rec["iters"], "fits": fits,
                         "median_recomputed": round(statistics.median(fits), 3) if fits else None,
                         "median_in_log": rec.get("median_fit_seconds"),
                         "system_cpu_busy_pct": [r.get("system_cpu_busy_mean_pct") for r in rec.get("reps", [])],
                         "gpu_mem_baseline_mib": [r.get("gpu_mem_baseline_mib") for r in rec.get("reps", [])]}
        except Exception as exc:
            out[name] = {"error": repr(exc)}
    g = out.get("gpu_retry", {}).get("median_recomputed")
    c = out.get("cpu_thread1", {}).get("median_recomputed")
    if g and c:
        out["claim_ratio_gpu_over_cpu1_recomputed"] = round(g / c, 4)
    print("CLAIM_RECOMPUTE " + json.dumps(out, indent=1), flush=True)


def summarize(results):
    def med(xs):
        return round(statistics.median(xs), 3) if xs else None

    summary = {}
    for phase in ("phase1_claim_size", "phase2_smoke_size"):
        rs = [r for r in results if r["phase"] == phase and r["error"] is None]
        devs = []
        for r in rs:
            if r["device"] not in devs:
                devs.append(r["device"])
        per = {}
        for dev in devs:
            ts = [r["fit_seconds"] for r in rs if r["device"] == dev]
            busy = [r["system_cpu_busy_pct"] for r in rs if r["device"] == dev]
            util = [r.get("gpu_util_mean_pct") for r in rs if r["device"] == dev]
            per[dev] = {"n": len(ts), "fits": ts, "median_fit_s": med(ts),
                        "min_fit_s": min(ts) if ts else None, "max_fit_s": max(ts) if ts else None,
                        "cv_pct": round(100 * statistics.pstdev(ts) / statistics.mean(ts), 1)
                        if len(ts) > 1 else None,
                        "system_cpu_busy_pct": busy, "gpu_util_mean_pct": util}
        by_key = {(r["round"], r["device"]): r["fit_seconds"] for r in rs}
        rounds = sorted(set(r["round"] for r in rs))
        ratios = {}
        for dev in devs:
            if dev == "GPU":
                continue
            ratios[f"GPU_over_{dev}_per_round"] = [
                round(by_key[(k, "GPU")] / by_key[(k, dev)], 3)
                for k in rounds if (k, "GPU") in by_key and (k, dev) in by_key]
        cpu_devs = [d for d in devs if d != "GPU"]
        if cpu_devs and "GPU" in per:
            best = min(cpu_devs, key=lambda d: per[d]["median_fit_s"])
            summary[phase] = {"per_device": per, "ratios": ratios,
                              "best_cpu_by_median": best,
                              "median_ratio_GPU_over_best_cpu": round(per["GPU"]["median_fit_s"] / per[best]["median_fit_s"], 3)}
        else:
            summary[phase] = {"per_device": per, "ratios": ratios}
    print("SUMMARY " + json.dumps(summary, indent=1), flush=True)


def main():
    meta = {"catboost": cb.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
            "gpu_ram_part": GPU_RAM_PART, "p1": P1, "p2": P2, "p1_rounds": P1_ROUNDS, "p2_rounds": P2_ROUNDS,
            "sampler_interval_s": 1.0, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("META " + json.dumps(meta), flush=True)
    claim_recompute()

    results = []
    sampler = GpuSampler()
    sampler.start()
    time.sleep(2.0)
    try:
        Xtr, ytr = make_data(P1["n_train"], P1["n_valid"], P1["n_feat"])
        print(f"DATA phase1 train shape {Xtr.shape} positive rate {ytr.mean():.4f}", flush=True)
        gpu_warmup(Xtr, ytr)
        for rnd in range(P1_ROUNDS):
            order = ["GPU", "CPU1"] if rnd % 2 == 0 else ["CPU1", "GPU"]
            for dev in order:
                results.append(timed_fit(dev, P1["iters"], Xtr, ytr, "phase1_claim_size", rnd, sampler))

        Xs, ys = make_data(P2["n_train"], P2["n_valid"], P2["n_feat"])
        print(f"DATA phase2 train shape {Xs.shape} positive rate {ys.mean():.4f}", flush=True)
        base = ["GPU", "CPU16", "CPU4", "CPU1"]
        for rnd in range(P2_ROUNDS):
            order = base[rnd:] + base[:rnd]
            for dev in order:
                results.append(timed_fit(dev, P2["iters"], Xs, ys, "phase2_smoke_size", rnd, sampler))
    finally:
        sampler.stop()
    summarize(results)
    print("DONE " + time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
