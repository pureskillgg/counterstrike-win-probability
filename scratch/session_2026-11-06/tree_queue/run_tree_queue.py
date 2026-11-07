"""Local tree queue on the sprint matrix (pin f9053b41, 6 Nov 2026). Validation only.

Reads only the sprint matrix (fit, early-stop and validation slices). No test or holdout rows exist in it.
Results: results/<id>.json (resumable: a finished cell is skipped), preds/<id>.parquet, boosters/f77_s0.json.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = "G:/pureskill/csds-cs2"
sys.path.insert(0, "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/experiments/sprint")
sys.path.insert(0, ROOT + "/scripts/model")
import sprint_worker as SW  # noqa: E402  (only for load_matrix and the GEO/BLIND group lists)
import wp_common as W  # noqa: E402

MDIR = "C:/Users/<user>/sprint/matrix_trainctr"
OUT = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/tree_queue"
for d in ("results", "preds", "boosters"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

meta, X, y, vk = SW.load_matrix(MDIR)
ALL = meta["feats_all"]
assert len(ALL) == 83, len(ALL)
n_fit, n_es = meta["n_fit"], meta["n_es"]
ARM = ["st_armor_sum_ct", "st_armor_sum_t"]
VIEW = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
GEO, BLIND = SW.GEO, SW.BLIND
HEAVY = ["st_heavy_ct", "st_heavy_t"]
SETS = {
    "f83": ALL,
    "f77": [f for f in ALL if f not in ARM and f not in VIEW],
    "noGEO": [f for f in ALL if f not in GEO],
    "noVIEW": [f for f in ALL if f not in VIEW],
    "noBLIND": [f for f in ALL if f not in BLIND],
    "noArmour": [f for f in ALL if f not in ARM],
    "noHeavy": [f for f in ALL if f not in HEAVY],
}
for k, v in SETS.items():
    assert all(f in ALL for f in v), k
print({k: len(v) for k, v in SETS.items()}, flush=True)

CELLS = [  # (id, set, seed, depth, eta, mcw)
    ("f77_s0", "f77", 0, 4, 0.03, 20),
    ("f77_s1", "f77", 1, 4, 0.03, 20),
    ("f77_s2", "f77", 2, 4, 0.03, 20),
    ("d3_77", "f77", 0, 3, 0.03, 20),
    ("d5_77", "f77", 0, 5, 0.03, 20),
    ("mcw50_77", "f77", 0, 4, 0.03, 50),
    ("abl_noGEO", "noGEO", 0, 4, 0.03, 20),
    ("abl_noVIEW", "noVIEW", 0, 4, 0.03, 20),
    ("abl_noBLIND", "noBLIND", 0, 4, 0.03, 20),
    ("abl_noArmour", "noArmour", 0, 4, 0.03, 20),
    ("abl_noHeavy", "noHeavy", 0, 4, 0.03, 20),
    ("f83_s0_check", "f83", 0, 4, 0.03, 20),
]


def ll(yv, p):
    return float(-(yv * np.log(np.clip(p, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1))).mean())


def prepare(feats):
    idx = [ALL.index(f) for f in feats]
    a1, a2 = n_fit, n_fit + n_es
    t0 = time.time()
    dtr = xgb.QuantileDMatrix(np.ascontiguousarray(X[:a1][:, idx]), np.asarray(y[:a1]), max_bin=256)
    dva = xgb.QuantileDMatrix(np.ascontiguousarray(X[a1:a2][:, idx]), np.asarray(y[a1:a2]), ref=dtr, max_bin=256)
    dvl = xgb.QuantileDMatrix(np.ascontiguousarray(X[a2:][:, idx]), np.asarray(y[a2:]), ref=dtr, max_bin=256)
    return {"feats": feats, "dtr": dtr, "dva": dva, "dvl": dvl, "yv": np.asarray(y[a2:]), "build_s": round(time.time() - t0, 1)}


def fit(prep, cid, seed, depth, eta, mcw):
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": eta, "max_depth": depth,
              "min_child_weight": mcw, "subsample": 0.8, "tree_method": "hist", "device": "cuda", "seed": seed,
              "max_bin": 256, "monotone_constraints": tuple(W.mono_for(prep["feats"]))}
    t0 = time.time()
    bst = xgb.train(params, prep["dtr"], num_boost_round=3000, evals=[(prep["dva"], "validation_0")],
                    early_stopping_rounds=100, verbose_eval=False)
    best = int(bst.best_iteration)
    fit_s = time.time() - t0
    p = bst.predict(prep["dvl"], iteration_range=(0, best + 1))
    res = {"id": cid, "status": "ok", "best_iteration": best, "val_logloss": ll(prep["yv"], p),
           "fit_seconds": round(fit_s, 1), "quantise_seconds": prep["build_s"], "n_features": len(prep["feats"]),
           "seed": seed, "depth": depth, "eta": eta, "mcw": mcw, "n_fit": int(n_fit), "n_early_stop": int(n_es),
           "n_validation": int(len(prep["yv"])), "test_rows_loaded": 0}
    vk.with_columns(pl.Series("p", p)).write_parquet(f"{OUT}/preds/{cid}.parquet")
    if cid == "f77_s0":
        bst.save_model(f"{OUT}/boosters/f77_s0.json")
        json.dump(prep["feats"], open(f"{OUT}/boosters/f77_s0_features.json", "w"))
    return res


by_set = {}
for c in CELLS:
    by_set.setdefault(c[1], []).append(c)
for s, cells in by_set.items():
    todo = [c for c in cells if not os.path.exists(f"{OUT}/results/{c[0]}.json")]
    if not todo:
        continue
    prep = prepare(SETS[s])
    print(f"set {s}: {len(SETS[s])} features, quantised in {prep['build_s']}s", flush=True)
    for cid, _, seed, depth, eta, mcw in todo:
        try:
            res = fit(prep, cid, seed, depth, eta, mcw)
        except Exception as e:  # keep the queue going; the error is recorded
            res = {"id": cid, "status": "error", "error": repr(e)[:500]}
        json.dump(res, open(f"{OUT}/results/{cid}.json", "w"), indent=1)
        print(json.dumps({k: res.get(k) for k in ("id", "status", "best_iteration", "val_logloss", "fit_seconds")}), flush=True)
    del prep
print("QUEUE_DONE", flush=True)
