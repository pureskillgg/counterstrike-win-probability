"""Debug: XGBoost monotone (+1 on x0) + isotonic CalibratedClassifierCV, cv=3. Synthetic data."""
import os
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


def true_logit(X):
    return 1.0 * X[:, 0] + 0.6 * np.sin(4.0 * X[:, 1]) * X[:, 2]


def make_data(n, seed):
    r = np.random.default_rng(seed)
    X = r.uniform(-1.0, 1.0, size=(n, 3))
    p = 1.0 / (1.0 + np.exp(-true_logit(X)))
    return X, (r.uniform(size=n) < p).astype(int)


def mono_scan(predict, X_bg, label):
    grid = np.linspace(-1.0, 1.0, 101)
    viol, max_drop, bad_rows = 0, 0.0, []
    for i, row in enumerate(X_bg):
        Z = np.tile(row, (101, 1))
        Z[:, 0] = grid
        p = predict(Z)
        d = np.diff(p)
        k = int(np.sum(d < -1e-12))
        if k:
            bad_rows.append(i)
        viol += k
        if d.min() < 0:
            max_drop = max(max_drop, float(-d.min()))
    print(f"{label:<42} violations={viol:<6} max_drop={max_drop:.3e} rows_with_violation={len(bad_rows)}", flush=True)
    return bad_rows


X, y = make_data(8000, seed=21)
X_tr, y_tr = X[:4000], y[:4000]
X_bg, _ = make_data(200, seed=1)

base = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                         tree_method="hist", n_jobs=2, random_state=0,
                         monotone_constraints=(1, 0, 0))
cal = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3).fit(X_tr, y_tr)

mono_scan(lambda Z: cal.predict_proba(Z)[:, 1], X_bg, "calibrated isotonic (ensemble, full)")
for k, cc in enumerate(cal.calibrated_classifiers_):
    est = cc.estimator
    mono_scan(lambda Z, est=est: est.predict_proba(Z)[:, 1], X_bg, f"fold {k} base xgb predict_proba")
    calibrator = cc.calibrators[0]
    grid_raw = np.linspace(0, 1, 2001)
    cal_out = calibrator.predict(grid_raw)
    print(f"fold {k} calibrator monotone in raw score: {bool(np.all(np.diff(cal_out) >= -1e-15))}", flush=True)
    print(f"fold {k} calibrator type: {type(calibrator).__name__}", flush=True)

# Check whether the violations come from raw scores that are NOT non-decreasing in x0 for the
# fold model: compare the raw base-model scan with the calibrated scan on the same rows.
bad = mono_scan(lambda Z: cal.predict_proba(Z)[:, 1], X_bg, "recheck calibrated")
if bad:
    i = bad[0]
    row = X_bg[i]
    Z = np.tile(row, (101, 1))
    Z[:, 0] = np.linspace(-1.0, 1.0, 101)
    p = cal.predict_proba(Z)[:, 1]
    d = np.diff(p)
    j = int(np.argmin(d))
    print("example row", i, "worst step index", j, "p[j]", p[j], "p[j+1]", p[j + 1], flush=True)
    for k, cc in enumerate(cal.calibrated_classifiers_):
        print(f"  fold {k} base raw at j, j+1:",
              cc.estimator.predict_proba(Z[[j, j + 1]])[:, 1], flush=True)
