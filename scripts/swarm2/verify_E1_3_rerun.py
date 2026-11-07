"""verify_E1_3, part 2: live CPU re-measurement with the PROJECT interpreter (G: env).

Why this script exists: E1 timed a 1M x 50 workload (200 trees) on a host at 100% CPU load and reported
CPU-default vs CPU-1-thread vs CUDA ratios. This script re-runs the two CPU configs on E1's synthetic generator
(same seed and feature recipe, copied from scripts/swarm2/E1_bench.py, not imported) to check
  (a) whether the CPU default-thread vs 1-thread ratio reproduces now, and
  (b) whether multi-threaded CPU timing is stable run to run.

Deviations, all ASSUMPTION and stated in the log:
  - 20 boosting rounds instead of 200 for the 1M-row runs (time budget and host load).
  - 3 interleaved reps per config (E1 had 1 run for default and 2 for 1-thread).
  - The CUDA leg is NOT re-timed: the project interpreter's xgboost is a CPU build (probed below).
Nothing is written to disk by this script; stdout goes to logs/swarm2/verify_E1_3.log.
"""

import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np
import xgboost as xgb

N_TRAIN = 1_000_000  # same as E1
N_HOLD = 100_000  # same as E1
N_FEAT = 50  # same as E1
SEED = 12345  # same as E1
ETA = 0.1  # same as E1
MAX_DEPTH = 4  # same as E1
WARM_ROUNDS = 5  # same as E1
TREES_BIG = 20  # ASSUMPTION: scaled down from E1's 200
REPS_BIG = 3  # ASSUMPTION: interleaved; E1 had 1 (default) and 2 (1-thread)
N_SMALL = 100_000  # same as E1_threads
TREES_SMALL = 10  # same as E1_threads
REPS_SMALL = 2  # ASSUMPTION: E1_threads also used 2 timed runs per config
SMALL_THREADS = (1, 2, 16)  # ASSUMPTION: subset of E1_threads' 1,2,4,8,16


def log(*args):
    print(*args, flush=True)


def host_snapshot(tag):
    """nvidia-smi memory/util and host CPU load, for contention context (same probes as E1)."""
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        gpu = f"nvidia-smi failed: {e}"
    try:
        cpu = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).LoadPercentage"],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        cpu = f"cpu probe failed: {e}"
    log(f"  [snap {tag}] gpu(mem,util)={gpu} | host_cpu_load%={cpu}")
    return cpu


def make_data():
    """Copied from scripts/swarm2/E1_bench.py make_data() so the data are identical to E1's."""
    rng = np.random.default_rng(SEED)
    n = N_TRAIN + N_HOLD
    X = rng.standard_normal((n, N_FEAT), dtype=np.float32)
    logit = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.5 * X[:, 2] * X[:, 3]
        + 0.4 * np.sin(2.0 * X[:, 4])
        + 0.3 * (X[:, 5] > 0)
        + 0.5 * rng.standard_normal(n).astype(np.float32)
    ).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def params(nthread):
    """Same parameters as E1_bench.py params_for(device='cpu', nthread)."""
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": MAX_DEPTH,
        "eta": ETA,
        "seed": 0,
    }
    if nthread is not None:
        p["nthread"] = nthread
    return p


def main():
    log("== PART 2: live CPU re-measurement with the project interpreter ==")
    log("start:", time.strftime("%Y-%m-%d %H:%M:%S"))
    log("python:", sys.executable)
    log("xgboost:", xgb.__version__, "numpy:", np.__version__)
    bi = xgb.build_info()
    log("build_info USE_CUDA =", bi.get("USE_CUDA"), "CUDA_VERSION =", bi.get("CUDA_VERSION"),
        "USE_OPENMP =", bi.get("USE_OPENMP"))
    log("logical_cpus:", os.cpu_count())

    # CUDA probe on tiny data. Expected to fail on a CPU-only build.
    Xp = np.random.default_rng(0).standard_normal((2000, 5)).astype(np.float32)
    yp = (Xp[:, 0] > 0).astype(np.float32)
    try:
        bst_p = xgb.train({"device": "cuda", "tree_method": "hist", "max_depth": 3},
                          xgb.DMatrix(Xp, label=yp), num_boost_round=2)
        cfg = json.loads(bst_p.save_config())
        log("cuda probe: train returned, saved device =", cfg["learner"]["generic_param"].get("device"))
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()
        log("cuda probe: FAILED ->", type(e).__name__, "|", (msg[0][:300] if msg else "(empty message)"))

    host_snapshot("start")

    t0 = time.perf_counter()
    Xtr, ytr, Xho, yho = make_data()
    log(f"data generated in {time.perf_counter() - t0:.2f} s; X_train {Xtr.shape} {Xtr.dtype} "
        f"{Xtr.nbytes / 2**20:.1f} MiB")
    t0 = time.perf_counter()
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dho = xgb.DMatrix(Xho)
    log(f"DMatrix build {time.perf_counter() - t0:.3f} s")

    # Untimed warm-ups. Default threads first, as in E1, so the one-time quantile-cut build is not timed.
    for nm, nth in (("default", None), ("t1", 1)):
        t0 = time.perf_counter()
        xgb.train(params(nth), dtr, num_boost_round=WARM_ROUNDS)
        log(f"warm-up {nm} ({WARM_ROUNDS} rounds, untimed): {time.perf_counter() - t0:.3f} s")

    log("")
    log(f"-- big: 1M x 50, {TREES_BIG} trees, interleaved t1/default, {REPS_BIG} reps --")
    big = {"t1": [], "default": []}
    first_pred = {}
    for rep in range(1, REPS_BIG + 1):
        for nm, nth in (("t1", 1), ("default", None)):
            host_snapshot(f"{nm} rep{rep} before")
            t0 = time.perf_counter()
            bst = xgb.train(params(nth), dtr, num_boost_round=TREES_BIG)
            el = time.perf_counter() - t0
            big[nm].append(el)
            p = bst.predict(dho)
            if rep == 1:
                first_pred[nm] = p
            log(f"  {nm:8s} rep{rep}: {el:.3f} s  ({el / TREES_BIG:.4f} s/tree)")
            host_snapshot(f"{nm} rep{rep} after")

    log("")
    for nm in ("t1", "default"):
        t = big[nm]
        log(f"{nm:8s} runs={[round(x, 3) for x in t]} median={statistics.median(t):.3f} s "
            f"s/tree={statistics.median(t) / TREES_BIG:.4f} max/min={max(t) / min(t):.3f}")
    pair = [big["default"][i] / big["t1"][i] for i in range(REPS_BIG)]
    log(f"per-rep default/t1 time ratio: {[round(r, 3) for r in pair]}  (>1 means default threads SLOWER)")
    log(f"median default/t1 = {statistics.median(big['default']) / statistics.median(big['t1']):.3f}")
    log(f"200-tree extrapolation (ASSUMPTION, linear in trees): "
        f"t1 {statistics.median(big['t1']) / TREES_BIG * 200:.1f} s, "
        f"default {statistics.median(big['default']) / TREES_BIG * 200:.1f} s")
    dmax = float(np.max(np.abs(first_pred["t1"].astype(np.float64) - first_pred["default"].astype(np.float64))))
    log(f"holdout pred max abs diff, t1 vs default (same 20-tree model, rep1): {dmax:.3g}")

    log("")
    log(f"-- small: 100k x 50, {TREES_SMALL} trees, nthread in {SMALL_THREADS}, {REPS_SMALL} reps interleaved --")
    ds = xgb.DMatrix(Xtr[:N_SMALL], label=ytr[:N_SMALL])
    xgb.train(params(1), ds, num_boost_round=2)  # untimed warm-up: builds cuts for this DMatrix
    small = {n: [] for n in SMALL_THREADS}
    for rep in range(1, REPS_SMALL + 1):
        for nth in SMALL_THREADS:
            t0 = time.perf_counter()
            xgb.train(params(nth), ds, num_boost_round=TREES_SMALL)
            el = time.perf_counter() - t0
            small[nth].append(el)
            log(f"  nthread={nth:2d} rep{rep}: {el:.3f} s ({el / TREES_SMALL:.4f} s/tree)")
    base = statistics.median(small[1])
    for nth in SMALL_THREADS:
        t = small[nth]
        log(f"nthread={nth:2d} runs={[round(x, 3) for x in t]} median={statistics.median(t):.3f} s "
            f"median/nthread1={statistics.median(t) / base:.2f}x")

    log("")
    log("CUDA leg: NOT re-timed here (see the cuda probe above). E1's CUDA runs are only checked from their log.")
    host_snapshot("end")
    log("end:", time.strftime("%Y-%m-%d %H:%M:%S"))
    log("PART 2 DONE")


if __name__ == "__main__":
    main()
