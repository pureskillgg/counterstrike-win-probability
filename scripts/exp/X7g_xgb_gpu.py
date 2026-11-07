"""X7g: GPU-accelerated stand-in for X7a. XGBoost hist on CUDA, leaf-wise growth capped at 15 leaves
(grow_policy lossguide), eta 0.05, monotone vector as baseline, early stopping 100 on validation, train-only fit.
Purpose: sklearn HistGradientBoosting has no GPU path, so this variant gives a GPU-accelerated run of the same
model family and tree size. min_child_weight 5 (hessian units, about 20 samples at p ~ 0.25) approximates HGB
min_samples_leaf 20. No row subsampling, to match HGB. Not the sklearn model.
Run with: <GPU env python> -I -X utf8 scripts\exp\X7g_xgb_gpu.py
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import sys
import time

import numpy as np

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # noqa: E402
import xgboost as xgb  # noqa: E402

PRED_DIR = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
ID = "X7g"
MONO = "(1,-1,1,0,0,0,0,1,0,0)"

t0 = time.time()
D = E.load()
dtr = xgb.DMatrix(D["train"]["X"].astype(np.float32), label=D["train"]["y"].astype(np.float32))
dva = xgb.DMatrix(D["val"]["X"].astype(np.float32), label=D["val"]["y"].astype(np.float32))
dte = xgb.DMatrix(D["test"]["X"].astype(np.float32))
yva = D["val"]["y"].astype(np.int64)
print("loaded", round(time.time() - t0, 1), "s", flush=True)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "grow_policy": "lossguide",
    "max_leaves": 15,
    "max_depth": 0,
    "eta": 0.05,
    "min_child_weight": 5.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 0.0,
    "max_bin": 256,
    "monotone_constraints": MONO,
    "seed": 0,
    "nthread": 4,
}

bst = None
for attempt in range(4):
    try:
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=3000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        break
    except xgb.core.XGBoostError as e:
        msg = str(e).lower()
        if "memory" in msg and attempt < 3:
            print("CUDA out-of-memory, waiting 30 s and retrying (attempt", attempt + 1, ")", flush=True)
            time.sleep(30)
            continue
        raise
assert bst is not None
best = int(bst.best_iteration)
print("best_iteration", best, "best val logloss", bst.best_score, flush=True)
print("device in saved config:", '"device":' in bst.save_config() and '"cuda"' in bst.save_config(), flush=True)

pv = bst.predict(dva, iteration_range=(0, best + 1)).astype(np.float64)
pt = bst.predict(dte, iteration_range=(0, best + 1)).astype(np.float64)
assert np.isfinite(pv).all() and np.isfinite(pt).all()
os.makedirs(PRED_DIR, exist_ok=True)
np.save(os.path.join(PRED_DIR, f"{ID}_val.npy"), pv)
np.save(os.path.join(PRED_DIR, f"{ID}_test.npy"), pt)
val_final = float(E.ll_vec(pv, yva).mean())
print("val logloss (saved preds)", round(val_final, 6), flush=True)

E.report(
    ID,
    D["test"],
    pt,
    extra={
        "model": "xgboost hist CUDA lossguide max_leaves=15 eta=0.05 mcw=5 monotone, ES 100 on val",
        "device": "cuda",
        "best_iter": best + 1,
        "val_logloss": val_final,
    },
)
print("total s", round(time.time() - t0, 1), flush=True)
