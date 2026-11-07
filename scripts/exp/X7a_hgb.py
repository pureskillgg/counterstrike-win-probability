"""X7a: sklearn HistGradientBoostingClassifier (CPU), 15 leaves, lr 0.05, monotone vector as baseline.

Early stopping: sklearn's built-in early stopping would carve an internal split out of train, so it is off.
One model is fit on the full train split for up to 800 iterations; validation log loss is computed at every
stage with staged_predict_proba and the best stage (argmin) is used for val and test. This is early stopping
with unlimited patience, selected on validation. The model is train-only (no train+val refit).
Subsampling (0.8) is not available in sklearn HGB and is omitted.
Run with: <project CPU python> -I -X utf8 scripts\exp\X7a_hgb.py
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import sys
import time

import numpy as np

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

PRED_DIR = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
ID = "X7a"
MONO = [1, -1, 1, 0, 0, 0, 0, 1, 0, 0]
MAX_ITER = 800

t0 = time.time()
D = E.load()
Xtr = D["train"]["X"].astype(np.float64)
ytr = D["train"]["y"].astype(np.int64)
Xva = D["val"]["X"].astype(np.float64)
yva = D["val"]["y"].astype(np.int64)
Xte = D["test"]["X"].astype(np.float64)
print("loaded", Xtr.shape, Xva.shape, Xte.shape, round(time.time() - t0, 1), "s", flush=True)

model = HistGradientBoostingClassifier(
    max_leaf_nodes=15,
    learning_rate=0.05,
    max_iter=MAX_ITER,
    early_stopping=False,
    monotonic_cst=MONO,
    min_samples_leaf=20,
    random_state=0,
)
model.fit(Xtr, ytr)
n_used = int(model.n_iter_)
print("fit done, n_iter_", n_used, round(time.time() - t0, 1), "s", flush=True)

# pass 1: validation log loss per stage
val_ll = np.empty(n_used)
for i, p in enumerate(model.staged_predict_proba(Xva)):
    val_ll[i] = float(E.ll_vec(p[:, 1], yva).mean())
best = int(np.argmin(val_ll))
print("best stage (trees)", best + 1, "val logloss", round(float(val_ll[best]), 6), flush=True)

# pass 2: take the best stage for val and test together
pv = pt = None
for i, (pa, pb) in enumerate(zip(model.staged_predict_proba(Xva), model.staged_predict_proba(Xte))):
    if i == best:
        pv = pa[:, 1].astype(np.float64).copy()
        pt = pb[:, 1].astype(np.float64).copy()
        break
assert pv is not None and pt is not None
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
        "model": "sklearn HistGradientBoostingClassifier max_leaf_nodes=15 lr=0.05 min_samples_leaf=20 monotone",
        "device": "cpu (sklearn HGB has no GPU path)",
        "best_iter": best + 1,
        "val_logloss": val_final,
    },
)
print("total s", round(time.time() - t0, 1), flush=True)
