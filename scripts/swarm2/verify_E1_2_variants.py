"""verify E1-2, stage C: two 100k-row variants of the CPU vs CUDA comparison (synthetic data, one seed).

(a) quantized : features rounded to a 0.25 grid (about 33-38 distinct values per feature), E1 params
                (200 trees, depth 4, eta 0.1, no subsampling). If the device gap is a sketch-approximation effect,
                it should shrink when max_bin is never reached. (ASSUMPTION about the mechanism.)
(b) bakeoff  : continuous features, params modelled on the XGBoost cell in scripts/model/bakeoff.py
                (max_depth 4, eta 0.05, min_child_weight 20, subsample 0.8, hist), fixed 300 trees.
                NOT replicated: n_estimators 3000 with early stopping, monotone_constraints, the real features.
                The seed is 0 (ASSUMPTION; bakeoff uses random_state=SEED).
Each variant: CPU nthread=1, CUDA, and a CPU row-permutation control (noise floor for this config).

Usage: <python> -I -X utf8 scripts/swarm2/verify_E1_2_variants.py
"""
import faulthandler
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb

faulthandler.enable()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_E1_2_core import compare, cut_cmp  # noqa: E402

SEED = 12345  # ASSUMPTION (same generator as E1_parity.py)
N_TRAIN = 100_000
N_HOLD = 100_000
N_FEAT = 50
E1_PARAMS = {"max_depth": 4, "eta": 0.1}
BAKEOFF_PARAMS = {"max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8}


def make_data():
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


def train(Xtr, ytr, device, extra, n_trees, nthread=None):
    p = {"objective": "binary:logistic", "tree_method": "hist", "device": device, "seed": 0}
    p.update(extra)
    if nthread is not None:
        p["nthread"] = nthread
    d = xgb.DMatrix(Xtr, label=ytr)
    t0 = time.perf_counter()
    b = xgb.train(p, d, num_boost_round=n_trees)
    dt = time.perf_counter() - t0
    cuts = None
    try:
        ptr, vals = d.get_quantile_cut()
        cuts = (np.asarray(ptr).copy(), np.asarray(vals, dtype=np.float64).copy())
    except Exception as e:  # noqa: BLE001
        print("  get_quantile_cut failed:", type(e).__name__, str(e)[:200], flush=True)
    del d
    return b, dt, cuts


def run_variant(tag, Xtr, ytr, Xho, yho, extra, n_trees):
    print(f"\n==================== variant {tag} ====================", flush=True)
    print("params", json.dumps({**extra, "n_trees": n_trees}), flush=True)
    dho = xgb.DMatrix(Xho)
    bc, tc, cc = train(Xtr, ytr, "cpu", extra, n_trees, nthread=1)
    pc = bc.predict(dho)
    bg, tg, cg = train(Xtr, ytr, "cuda", extra, n_trees)
    pg = bg.predict(dho)
    print(f"train times: cpu {tc:.1f} s, cuda {tg:.1f} s", flush=True)
    ref = compare(f"{tag}: cpu_vs_cuda", pc, pg, yho)
    print(f"{tag}: cut cpu vs cuda:", cut_cmp(cc, cg), flush=True)
    out = {"ref": ref, "cut": cut_cmp(cc, cg)}
    for k in (1, 2):
        perm = np.random.default_rng(4000 + k).permutation(len(ytr))
        bp, _, cp = train(Xtr[perm], ytr[perm], "cpu", extra, n_trees, nthread=1)
        pp = bp.predict(dho)
        r = compare(f"{tag}: noise_cpu_perm{k}_vs_cpu_orig", pc, pp, yho)
        out[f"noise_perm{k}"] = r
        print(f"{tag}: cut cpu_orig vs cpu_perm{k}:", cut_cmp(cc, cp), flush=True)
    return out


if __name__ == "__main__":
    print("xgboost", xgb.__version__, "USE_CUDA", xgb.build_info().get("USE_CUDA"), flush=True)
    Xtr, ytr, Xho, yho = make_data()
    distinct = [len(np.unique(Xtr[:, j])) for j in range(N_FEAT)]
    print("continuous distinct values per feature min/max:", min(distinct), max(distinct), flush=True)

    Xq = np.round(Xtr * 4.0) / 4.0
    Xqh = np.round(Xho * 4.0) / 4.0
    qd = [len(np.unique(Xq[:, j])) for j in range(N_FEAT)]
    print("quantized distinct values per feature min/max:", min(qd), max(qd), flush=True)
    res = {}
    res["quantized"] = run_variant("quantized", Xq, ytr, Xqh, yho, E1_PARAMS, 200)
    del Xq, Xqh
    res["bakeoff_like"] = run_variant("bakeoff_like", Xtr, ytr, Xho, yho, BAKEOFF_PARAMS, 300)

    print("\nVARIANTS_JSON_BEGIN", flush=True)
    print(json.dumps(res, indent=1, default=str), flush=True)
    print("VARIANTS_JSON_END", flush=True)
