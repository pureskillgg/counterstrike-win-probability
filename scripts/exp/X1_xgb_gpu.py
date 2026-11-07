"""X1: XGBoost on GPU (CUDA), same settings as the CPU baseline. Real-data check of CUDA vs CPU numerics."""
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")

import numpy as np
import xgboost as xgb
from pathlib import Path

import pt_eval as E

ID = "X1"
PRED_DIR = Path(r"G:\pureskill\csds-cs2\data\viz\pt_preds")
PRED_DIR.mkdir(parents=True, exist_ok=True)
MONO = (1, -1, 1, 0, 0, 0, 0, 1, 0, 0)

print("xgboost", xgb.__version__, "numpy", np.__version__, flush=True)
D = E.load()
Xtr, ytr = D["train"]["X"], D["train"]["y"]
Xva, yva = D["val"]["X"], D["val"]["y"]
Xte, yte = D["test"]["X"], D["test"]["y"]
print("shapes train", Xtr.shape, Xtr.dtype, "val", Xva.shape, "test", Xte.shape, flush=True)


def make_model():
    return xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        tree_method="hist",
        device="cuda",
        monotone_constraints=MONO,
        early_stopping_rounds=100,
        eval_metric="logloss",
        random_state=0,
    )


def fit_with_retry(max_tries=4):
    last = None
    for attempt in range(1, max_tries + 1):
        try:
            m = make_model()
            t0 = time.time()
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return m, time.time() - t0
        except xgb.core.XGBoostError as e:
            last = e
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                print(f"CUDA OOM on attempt {attempt}: {e}; waiting 30 s", flush=True)
                time.sleep(30)
                continue
            raise
    raise last


model, fit_sec = fit_with_retry()
best = int(model.best_iteration)
print(f"fit seconds {fit_sec:.2f} best_iteration {best} best_score {model.best_score}", flush=True)

rng = (0, best + 1)
pv = model.predict_proba(Xva, iteration_range=rng)[:, 1].astype(np.float64)
pt = model.predict_proba(Xte, iteration_range=rng)[:, 1].astype(np.float64)

np.save(PRED_DIR / f"{ID}_val.npy", pv)
np.save(PRED_DIR / f"{ID}_test.npy", pt)
print("saved", PRED_DIR / f"{ID}_val.npy", PRED_DIR / f"{ID}_test.npy", flush=True)

val_ll = float(np.mean(E.ll_vec(pv, yva)))
print(f"val logloss (pooled, for reference only) {val_ll:.6f}", flush=True)

# timing of the predict call alone is not needed; report fit time and best iteration
out = E.report(
    ID,
    D["test"],
    pt,
    extra={
        "device": "cuda",
        "fit_seconds": round(fit_sec, 2),
        "best_iteration": best,
        "val_logloss_pooled": val_ll,
        "xgboost_version": xgb.__version__,
        "params": "n_estimators=3000 lr=0.05 depth=4 mcw=20 subsample=0.8 hist cuda mono=baseline es=100 seed=0",
    },
)
print("DONE", ID, flush=True)
