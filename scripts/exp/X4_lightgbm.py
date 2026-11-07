"""X4: LightGBM (CPU) small grid on the per-second win-probability cache.

Run with the project CPU env (LightGBM 4.7.0):
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\exp\\X4_lightgbm.py

Grid: num_leaves {15,31,63} x learning_rate {0.03,0.05} x min_child_samples {50,200} x feature_fraction {0.8,1.0}.
Selection by validation log loss (early stopping 100 on val). Test scored once with the chosen model.
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # noqa: E402  (sets MKL_THREADING_LAYER too)
import lightgbm as lgb  # noqa: E402

T0 = time.time()
BUDGET_START_NEW_S = 400.0  # do not start a new grid config after this many seconds
NUM_THREADS = 4
MONO = [1, -1, 1, 0, 0, 0, 0, 1, 0, 0]
PRED_DIR = Path(r"G:\pureskill\csds-cs2\data\viz\pt_preds")
PRED_DIR.mkdir(parents=True, exist_ok=True)
ID = "X4"

D = E.load()
Xtr = np.ascontiguousarray(D["train"]["X"], dtype=np.float64)
ytr = np.ascontiguousarray(D["train"]["y"], dtype=np.float64)
Xva = np.ascontiguousarray(D["val"]["X"], dtype=np.float64)
yva = np.ascontiguousarray(D["val"]["y"], dtype=np.float64)
Xte = np.ascontiguousarray(D["test"]["X"], dtype=np.float64)
print(f"shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape} load_s={time.time()-T0:.1f}", flush=True)
assert Xtr.shape[1] == len(MONO) == Xva.shape[1] == Xte.shape[1] == 10

ds_params = {
    "max_bin": 255,
    "verbose": -1,
    "num_threads": NUM_THREADS,
    "monotone_constraints": MONO,
    "monotone_constraints_method": "basic",
}
dtr = lgb.Dataset(Xtr, label=ytr, params=ds_params, free_raw_data=False)
dtr.construct()
dva = lgb.Dataset(Xva, label=yva, reference=dtr, params=ds_params, free_raw_data=False)
print(f"dataset constructed s={time.time()-T0:.1f}", flush=True)

base = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "num_threads": NUM_THREADS,
    "monotone_constraints": MONO,
    "monotone_constraints_method": "basic",
    "seed": 0,
    "deterministic": True,
}

grid = list(itertools.product([15, 31, 63], [0.03, 0.05], [50, 200], [0.8, 1.0]))
print(f"grid size={len(grid)}", flush=True)

best = None
results = []
for i, (nl, lr, mcs, ff) in enumerate(grid):
    if time.time() - T0 > BUDGET_START_NEW_S:
        print(f"time budget reached before config {i}; grid stopped early", flush=True)
        break
    params = dict(base, num_leaves=nl, learning_rate=lr, min_child_samples=mcs, feature_fraction=ff)
    t = time.time()
    bst = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dva],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    bi = bst.best_iteration if bst.best_iteration and bst.best_iteration > 0 else bst.current_iteration()
    pva = bst.predict(Xva, num_iteration=bi)
    vll = float(E.ll_vec(pva, yva).mean())
    rec = {
        "config": i,
        "num_leaves": nl,
        "learning_rate": lr,
        "min_child_samples": mcs,
        "feature_fraction": ff,
        "best_iter": int(bi),
        "val_logloss": vll,
        "secs": round(time.time() - t, 1),
    }
    results.append(rec)
    print(json.dumps(rec), flush=True)
    if best is None or vll < best["rec"]["val_logloss"]:
        best = {"rec": rec, "bst": bst, "pva": pva, "bi": bi}
    else:
        del bst, pva

print(f"configs run={len(results)} total_grid_s={time.time()-T0:.1f}", flush=True)
assert best is not None, "no config completed"

chosen = best["rec"]
bst = best["bst"]
bi = best["bi"]
pva = np.asarray(best["pva"], dtype=np.float64)
pte = np.asarray(bst.predict(Xte, num_iteration=bi), dtype=np.float64)
assert pva.shape == (len(yva),) and pte.shape == (len(D["test"]["y"]),)
assert np.isfinite(pva).all() and np.isfinite(pte).all()

np.save(PRED_DIR / f"{ID}_val.npy", pva)
np.save(PRED_DIR / f"{ID}_test.npy", pte)
print(f"saved preds to {PRED_DIR}", flush=True)
print("chosen by val:", json.dumps(chosen), flush=True)

extra = {
    "device": "cpu",
    "lightgbm_version": lgb.__version__,
    "chosen": {k: chosen[k] for k in ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction")},
    "best_iter": int(bi),
    "val_logloss": chosen["val_logloss"],
    "configs_run": len(results),
    "grid_size": len(grid),
}
out = E.report(ID, D["test"], pte, extra=extra)
print(
    "RESULT "
    + json.dumps(
        {
            "test_logloss": out["logloss"],
            "ci95": out["logloss_ci95"],
            "vs_baseline": out["vs_cpu_xgb_baseline"],
            "vs_baseline_ci95": out["vs_cpu_xgb_baseline_ci95"],
            "seconds_total": round(time.time() - T0, 1),
        }
    ),
    flush=True,
)
