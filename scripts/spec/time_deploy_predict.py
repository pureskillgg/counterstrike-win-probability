"""Deployment timing and pure-Python parity check for the XGBoost baseline.

Read-only on models/. Writes only to notes/spec/deployment_timing_2026-10-13.json.
Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\spec\\time_deploy_predict.py
Synthetic inputs only: values follow the feature definitions in notes/model/MODEL_CARD.md
but are random, so timings and parity say nothing about accuracy.
"""
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(r"G:\pureskill\csds-cs2")
MODEL = ROOT / "models" / "xgb_primary_seed0.json"
META = ROOT / "models" / "xgb_primary_seed0_meta.json"
OUT = ROOT / "notes" / "spec" / "deployment_timing_2026-10-13.json"

N_ROWS = 10_000
BATCH_REPEATS = 30
PY_BATCH_REPEATS = 3
SINGLE_CALLS = 2000
SINGLE_DM_CALLS = 500

meta = json.loads(META.read_text(encoding="utf-8"))
FEATURES = meta["features"]
ITER_RANGE = tuple(meta["predict_iteration_range"])
N_USED = ITER_RANGE[1] - ITER_RANGE[0]
assert len(FEATURES) == 10


def synth(n, seed=0):
    rng = np.random.default_rng(seed)
    alive_ct = rng.integers(0, 6, n)
    alive_t = rng.integers(0, 6, n)
    planted = rng.integers(0, 2, n)
    since_live = rng.uniform(0, 180, n)
    since_plant = np.where(planted == 1, rng.uniform(0, 40, n), -1.0)
    round_no = rng.integers(1, 41, n)
    score_diff = rng.integers(-12, 13, n)
    is_pistol = np.isin(round_no, [1, 13]).astype(np.int64)
    is_ot = (round_no > 24).astype(np.int64)
    X = np.column_stack([
        alive_ct, alive_t, alive_ct - alive_t, planted, since_live,
        since_plant, round_no, score_diff, is_pistol, is_ot,
    ]).astype(np.float32)
    return X


def pct(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(math.ceil(p / 100 * len(xs))) - 1))
    return xs[k]


def load_py_trees(path, n_used):
    d = json.loads(path.read_text(encoding="utf-8"))
    gb = d["learner"]["gradient_booster"]
    all_trees = gb["model"]["trees"]
    depths = []
    node_counts = []
    for t in all_trees:
        node_counts.append(len(t["left_children"]))
    used = []
    for t in all_trees[:n_used]:
        used.append({
            "left": t["left_children"],
            "right": t["right_children"],
            "feat": t["split_indices"],
            "cond": t["split_conditions"],
            "dl": [bool(v) for v in t["default_left"]],
        })
    # max depth over used trees
    def depth_of(t):
        best = 0
        stack = [(0, 0)]
        while stack:
            node, dep = stack.pop()
            if t["left"][node] == -1:
                best = max(best, dep)
            else:
                stack.append((t["left"][node], dep + 1))
                stack.append((t["right"][node], dep + 1))
        return best
    max_depth_used = max(depth_of(t) for t in used)
    bs_raw = d["learner"]["learner_model_param"]["base_score"]
    bs = float(str(bs_raw).strip("[]"))
    base_margin = math.log(bs / (1 - bs))
    info = {
        "trees_in_json": len(all_trees),
        "trees_used": len(used),
        "max_nodes_per_tree_json": max(node_counts),
        "max_depth_used": max_depth_used,
        "base_score_raw": bs_raw,
        "base_margin": base_margin,
    }
    return used, base_margin, info


def predict_margin_py(trees, base_margin, x):
    m = base_margin
    for t in trees:
        L = t["left"]; R = t["right"]; F = t["feat"]; C = t["cond"]; DL = t["dl"]
        node = 0
        while L[node] != -1:
            v = x[F[node]]
            if v != v:
                node = L[node] if DL[node] else R[node]
            elif v < C[node]:
                node = L[node]
            else:
                node = R[node]
        m += C[node]
    return m


def main():
    result = {
        "created": "2026-10-13",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "xgboost_version": xgb.__version__,
        "numpy_version": np.__version__,
        "n_rows": N_ROWS,
        "predict_iteration_range": list(ITER_RANGE),
        "model_file_bytes": MODEL.stat().st_size,
        "features": FEATURES,
    }

    booster = xgb.Booster()
    booster.load_model(str(MODEL))
    booster.feature_names = FEATURES

    X = synth(N_ROWS, seed=0)

    # Parity and pure-Python path first (correctness before timing).
    trees, base_margin, tinfo = load_py_trees(MODEL, N_USED)
    result["tree_info"] = tinfo

    dm = xgb.DMatrix(X, feature_names=FEATURES, missing=np.nan)
    ref_margin = booster.predict(dm, output_margin=True, iteration_range=ITER_RANGE)
    ref_prob = booster.predict(dm, iteration_range=ITER_RANGE)

    py_margin = np.array([predict_margin_py(trees, base_margin, row) for row in X.tolist()])
    py_prob = 1 / (1 + np.exp(-py_margin))
    result["parity"] = {
        "max_abs_margin_diff": float(np.max(np.abs(py_margin - ref_margin))),
        "max_abs_prob_diff": float(np.max(np.abs(py_prob - ref_prob))),
        "n_rows_compared": int(N_ROWS),
    }

    # NaN handling check: a row with NaN in one feature should not crash and should match.
    Xn = X[:5].copy()
    Xn[:, 4] = np.nan
    dmn = xgb.DMatrix(Xn, feature_names=FEATURES, missing=np.nan)
    ref_nan = booster.predict(dmn, iteration_range=ITER_RANGE)
    py_nan = np.array([1 / (1 + math.exp(-predict_margin_py(trees, base_margin, row)))
                       for row in Xn.tolist()])
    result["nan_parity_max_abs_prob_diff"] = float(np.max(np.abs(py_nan - ref_nan)))

    # Timing 1: Booster batch, DMatrix rebuilt each call (the realistic serving path).
    times_dm = []
    for _ in range(BATCH_REPEATS):
        t0 = time.perf_counter()
        d = xgb.DMatrix(X, feature_names=FEATURES, missing=np.nan)
        booster.predict(d, iteration_range=ITER_RANGE)
        times_dm.append(time.perf_counter() - t0)

    # Timing 2: Booster batch, prebuilt DMatrix, predict only. Each DMatrix is new and
    # predicted once, so XGBoost's per-DMatrix prediction cache is not hit. (Re-predicting
    # the same DMatrix hits the cache and gives a misleading ~0.2 ms.)
    dms = [xgb.DMatrix(X, feature_names=FEATURES, missing=np.nan) for _ in range(BATCH_REPEATS)]
    times_pred = []
    for d_i in dms:
        t0 = time.perf_counter()
        booster.predict(d_i, iteration_range=ITER_RANGE)
        times_pred.append(time.perf_counter() - t0)

    # Timing 3: inplace_predict on numpy (no DMatrix).
    times_inplace = []
    for _ in range(BATCH_REPEATS):
        t0 = time.perf_counter()
        booster.inplace_predict(X, iteration_range=ITER_RANGE, missing=np.nan)
        times_inplace.append(time.perf_counter() - t0)

    # Timing 4: single-state latency, single thread, inplace_predict.
    booster.set_param({"nthread": 1})
    row = X[:1].copy()
    lat_inplace = []
    for _ in range(SINGLE_CALLS):
        t0 = time.perf_counter()
        booster.inplace_predict(row, iteration_range=ITER_RANGE, missing=np.nan)
        lat_inplace.append(time.perf_counter() - t0)

    # Timing 5: single-state latency, single thread, DMatrix per call.
    lat_dm = []
    for _ in range(SINGLE_DM_CALLS):
        t0 = time.perf_counter()
        d1 = xgb.DMatrix(row, feature_names=FEATURES, missing=np.nan)
        booster.predict(d1, iteration_range=ITER_RANGE)
        lat_dm.append(time.perf_counter() - t0)

    # Timing 6: pure-Python tree walk, batch of 10,000 (single thread by nature).
    rows_list = X.tolist()
    times_py = []
    for _ in range(PY_BATCH_REPEATS):
        t0 = time.perf_counter()
        for r in rows_list:
            predict_margin_py(trees, base_margin, r)
        times_py.append(time.perf_counter() - t0)
    lat_py = []
    for r in rows_list[:2000]:
        t0 = time.perf_counter()
        predict_margin_py(trees, base_margin, r)
        lat_py.append(time.perf_counter() - t0)

    def ms_stats(xs):
        return {
            "median_ms": round(statistics.median(xs) * 1000, 4),
            "p95_ms": round(pct(xs, 95) * 1000, 4),
            "min_ms": round(min(xs) * 1000, 4),
            "n": len(xs),
        }

    result["timing_ms"] = {
        "booster_batch10k_dmatrix_rebuilt_all_threads": ms_stats(times_dm),
        "booster_batch10k_prebuilt_dmatrix_all_threads": ms_stats(times_pred),
        "booster_batch10k_inplace_all_threads": ms_stats(times_inplace),
        "booster_single_row_inplace_1thread": ms_stats(lat_inplace),
        "booster_single_row_dmatrix_1thread": ms_stats(lat_dm),
        "pure_python_batch10k_single_thread": ms_stats(times_py),
        "pure_python_single_row_single_thread": ms_stats(lat_py),
    }

    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
