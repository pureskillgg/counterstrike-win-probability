"""GBDT verification (synthetic data only, under 10k rows).

Libraries: xgboost 3.4.2, lightgbm 4.7.0, sklearn 1.9.1 HistGradientBoostingClassifier.

Checks:
  1. monotone constraint (+1 on feature 0) with a real fit and a grid check,
     plus unconstrained controls so the check is shown to be discriminative
  2. sample_weight, early stopping on an explicit validation split,
     native missing values, categorical features
  3. categorical features combined with a monotone constraint
  4. CalibratedClassifierCV (sigmoid and isotonic, cv=3) with each estimator

Run:
  cd G:/pureskill/csds-cs2
  pixi run python -I scripts/research/gbdt_checks.py
"""
import os

# Keep CPU use small so the running builds are not starved.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import inspect
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

SEED = 0
GRID_N = 101
N_BACKGROUND = 200
LIBS = ("xgboost", "lightgbm", "hgb")
LEVELS = ["a", "b", "c", "d", "e"]
RESULTS = {}
PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lgb_cat_monotone_probe.py")


def banner(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def attempt(fn):
    """Run fn; on failure return the error text instead of stopping the script."""
    try:
        return fn()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def find_key(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            hit = find_key(v, key)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find_key(v, key)
            if hit is not None:
                return hit
    return None


def true_logit(X):
    return 1.0 * X[:, 0] + 0.6 * np.sin(4.0 * X[:, 1]) * X[:, 2]


def make_data(n, seed):
    """Binary target whose true logit increases in feature 0 (slope 1.0)."""
    r = np.random.default_rng(seed)
    X = r.uniform(-1.0, 1.0, size=(n, 3))
    p = 1.0 / (1.0 + np.exp(-true_logit(X)))
    y = (r.uniform(size=n) < p).astype(int)
    return X, y


def build(name, mono=False, categorical=False):
    """Return an unfitted estimator. mono=True puts +1 on feature 0 (first column)."""
    if name == "xgboost":
        kw = dict(n_estimators=200, max_depth=4, learning_rate=0.1,
                  tree_method="hist", n_jobs=2, random_state=SEED)
        if categorical:
            kw["enable_categorical"] = True
        if mono:
            kw["monotone_constraints"] = (1, 0, 0)
        return xgb.XGBClassifier(**kw)
    if name == "lightgbm":
        kw = dict(n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.1,
                  n_jobs=2, random_state=SEED, verbose=-1)
        if mono:
            kw["monotone_constraints"] = [1, 0, 0]
        return lgb.LGBMClassifier(**kw)
    kw = dict(max_iter=200, max_depth=4, learning_rate=0.1,
              early_stopping=False, random_state=SEED)
    if mono:
        kw["monotonic_cst"] = [1, 0, 0]
    if categorical:
        kw["categorical_features"] = "from_dtype"
    return HistGradientBoostingClassifier(**kw)


def monotone_violations(model, X_bg, col=0):
    """Count grid steps where P(y=1) drops as feature `col` increases (full grid over [-1, 1])."""
    grid = np.linspace(-1.0, 1.0, GRID_N)
    violations = 0
    worst_drop = 0.0
    for row in X_bg:
        Z = np.tile(row, (GRID_N, 1))
        Z[:, col] = grid
        p = model.predict_proba(Z)[:, 1]
        d = np.diff(p)
        violations += int(np.sum(d < -1e-12))
        worst_drop = max(worst_drop, float(-d.min()))
    return violations, worst_drop


def config_value(name, est):
    """Read back the monotone setting the library actually holds after fit."""
    if name == "xgboost":
        cfg = json.loads(est.get_booster().save_config())
        return find_key(cfg, "monotone_constraints")
    if name == "lightgbm":
        return est.booster_.dump_model().get("monotone_constraints")
    return None if est.monotonic_cst is None else list(est.monotonic_cst)


def metrics(y, p):
    return {"logloss": round(float(log_loss(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4)}


# ---------------------------------------------------------------- section 1-3
def monotone_section():
    banner("1-3. Monotone constraint on feature 0 (+1): signatures and real-fit grid check")
    sig = {}
    for label, cls, pname in (("xgboost", xgb.XGBClassifier, "monotone_constraints"),
                              ("lightgbm", lgb.LGBMClassifier, "monotone_constraints"),
                              ("hgb", HistGradientBoostingClassifier, "monotonic_cst")):
        params = inspect.signature(cls.__init__).parameters
        sig[label] = {"explicit_param": pname in params,
                      "var_kwargs": any(p.kind == p.VAR_KEYWORD for p in params.values())}
    print("signatures:", json.dumps(sig))
    RESULTS["signature"] = sig

    X_bg, _ = make_data(N_BACKGROUND, seed=SEED + 1)
    mono = {}
    for n in (500, 4000):
        X_tr, y_tr = make_data(n, seed=SEED)
        for name in LIBS:
            for constrained in (True, False):
                key = f"n={n} {'constrained' if constrained else 'unconstrained'} {name}"
                est = build(name, mono=constrained)
                try:
                    est.fit(X_tr, y_tr)
                except Exception as exc:
                    mono[key] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"{key:<44} FIT ERROR {exc}")
                    continue
                v, worst = monotone_violations(est, X_bg)
                cfg = config_value(name, est)
                mono[key] = {"violations": v, "worst_drop": round(worst, 6), "config_in_model": str(cfg)}
                print(f"{key:<44} violations={v:<6} worst_drop={worst:.6f} config_in_model={cfg}", flush=True)
    RESULTS["monotone"] = mono


def feature_section():
    banner("4. sample_weight, early stopping (explicit validation split), native missing, categorical")
    X_all, y_all = make_data(5000, seed=SEED + 2)
    X_tr, y_tr = X_all[:3000], y_all[:3000]
    X_va, y_va = X_all[3000:4000], y_all[3000:4000]
    X_te, y_te = X_all[4000:], y_all[4000:]

    # sample_weight: positives weighted 5x, so mean predicted probability should rise.
    w = np.where(y_tr == 1, 5.0, 1.0)
    sw = {}
    for name in LIBS:
        def run(name=name):
            plain = build(name).fit(X_tr, y_tr).predict_proba(X_te)[:, 1].mean()
            weighted = build(name).fit(X_tr, y_tr, sample_weight=w).predict_proba(X_te)[:, 1].mean()
            return {"mean_p_plain": round(float(plain), 4),
                    "mean_p_weighted": round(float(weighted), 4),
                    "weighted_higher": bool(weighted > plain)}
        sw[name] = attempt(run)
        print("sample_weight", name, sw[name], flush=True)
    RESULTS["sample_weight"] = sw

    # early stopping on the user's own validation split.
    def xgb_es():
        m = xgb.XGBClassifier(n_estimators=2000, max_depth=4, learning_rate=0.05,
                              tree_method="hist", n_jobs=2, random_state=SEED,
                              early_stopping_rounds=25, eval_metric="logloss")
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return {"best_iteration": int(m.best_iteration),
                "rounds_trained": int(m.get_booster().num_boosted_rounds())}

    def lgb_es():
        m = lgb.LGBMClassifier(n_estimators=2000, max_depth=4, num_leaves=15, learning_rate=0.05,
                               n_jobs=2, random_state=SEED, verbose=-1)
        m.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(25, verbose=False)])
        return {"best_iteration": int(m.best_iteration_),
                "trees_in_booster": int(m.booster_.num_trees())}

    def hgb_es():
        m = HistGradientBoostingClassifier(max_iter=2000, max_depth=4, learning_rate=0.05,
                                           early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=25, random_state=SEED)
        m.fit(X_tr, y_tr)  # internal split only; the user's X_va is not used
        try:
            HistGradientBoostingClassifier(max_iter=5).fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            eval_set_arg = "accepted"
        except TypeError as exc:
            eval_set_arg = f"TypeError: {exc}"
        return {"n_iter": int(m.n_iter_), "validation": "internal random split (validation_fraction=0.1)",
                "explicit_eval_set_arg": eval_set_arg}

    es = {"xgboost": attempt(xgb_es), "lightgbm": attempt(lgb_es), "hgb": attempt(hgb_es)}
    for k, v in es.items():
        print("early_stopping", k, v, flush=True)
    RESULTS["early_stopping"] = es

    # native missing values: 20% NaN in feature 1 in train and test.
    rng = np.random.default_rng(SEED + 7)
    Xm = X_tr.copy()
    Xm[rng.uniform(size=len(Xm)) < 0.2, 1] = np.nan
    Xtm = X_te.copy()
    Xtm[rng.uniform(size=len(Xtm)) < 0.2, 1] = np.nan
    miss = {}
    for name in LIBS:
        def run(name=name):
            p = build(name).fit(Xm, y_tr).predict_proba(Xtm)[:, 1]
            return {"nan_train": int(np.isnan(Xm).sum()), "nan_test": int(np.isnan(Xtm).sum()),
                    "all_predictions_finite": bool(np.isfinite(p).all()),
                    "logloss": round(float(log_loss(y_te, p)), 4)}
        miss[name] = attempt(run)
        print("native_missing", name, miss[name], flush=True)
    RESULTS["native_missing"] = miss

    # categorical features: effect of category on the logit is strong, so dropping it should hurt.
    r = np.random.default_rng(SEED + 11)
    n = 5000
    x0 = r.uniform(-1, 1, n)
    x1 = r.uniform(-1, 1, n)
    cat = r.choice(LEVELS, size=n)
    eff = {"a": -1.5, "b": -0.5, "c": 0.0, "d": 0.5, "e": 1.5}
    logit = x0 + 0.5 * x1 + np.array([eff[v] for v in cat])
    yc = (r.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({"x0": x0, "x1": x1, "cat": pd.Categorical(cat, categories=LEVELS)})
    Dtr, ytr = df.iloc[:3500], yc[:3500]
    Dte, yte = df.iloc[3500:], yc[3500:]
    catres = {}
    for name in LIBS:
        def run(name=name):
            out = {}
            for label, cols in (("with_cat", ["x0", "x1", "cat"]), ("without_cat", ["x0", "x1"])):
                m = build(name, categorical=True).fit(Dtr[cols], ytr)
                p = m.predict_proba(Dte[cols])[:, 1]
                out[label] = round(float(log_loss(yte, p)), 4)
            return out
        catres[name] = attempt(run)
        print("categorical_logloss", name, catres[name], flush=True)
    RESULTS["categorical"] = catres
    return Dtr, ytr, Dte, yte


def cat_monotone_section(Dtr, ytr):
    banner("3b. Monotone +1 on x0 with a categorical column present, and a constraint placed on the categorical")
    r = np.random.default_rng(SEED + 13)
    out = {}
    for name in LIBS:
        def run(name=name):
            m = build(name, mono=True, categorical=True).fit(Dtr[["x0", "x1", "cat"]], ytr)
            viol = 0
            worst = 0.0
            for _ in range(50):
                x1v = r.uniform(-1, 1)
                catv = r.choice(LEVELS)
                grid = np.linspace(-1.0, 1.0, GRID_N)
                Z = pd.DataFrame({"x0": grid, "x1": np.full(GRID_N, x1v),
                                  "cat": pd.Categorical([catv] * GRID_N, categories=LEVELS)})
                d = np.diff(m.predict_proba(Z[["x0", "x1", "cat"]])[:, 1])
                viol += int(np.sum(d < -1e-12))
                worst = max(worst, float(-d.min()))
            return {"x0_violations_with_cat": viol, "worst_drop": round(worst, 6)}

        def cat_constraint(name=name):
            if name == "lightgbm":
                # LightGBM aborts the process on this case (Fatal, not a Python exception),
                # so it runs in its own interpreter and the exit code is recorded.
                proc = subprocess.run([sys.executable, "-I", PROBE, "cat"],
                                      capture_output=True, text=True, timeout=300)
                lines = (proc.stdout + proc.stderr).splitlines()
                return {"returncode": proc.returncode,
                        "fatal_line": [s for s in lines if "Fatal" in s][:1],
                        "fit_returned_ok": any("fit returned OK" in s for s in lines)}
            if name == "xgboost":
                m = xgb.XGBClassifier(n_estimators=50, tree_method="hist", enable_categorical=True,
                                      monotone_constraints=(0, 0, 1), n_jobs=2)
            else:
                m = HistGradientBoostingClassifier(max_iter=50, monotonic_cst=[0, 0, 1],
                                                   categorical_features="from_dtype")
            m.fit(Dtr[["x0", "x1", "cat"]], ytr)
            return "accepted"

        out[name] = {"x0_constraint": attempt(run), "constraint_on_categorical": attempt(cat_constraint)}
        print("cat+monotone", name, out[name], flush=True)
    RESULTS["cat_monotone"] = out


# ---------------------------------------------------------------- section 6
def calibration_section():
    banner("6. CalibratedClassifierCV (cv=3) with each estimator: sigmoid vs isotonic")
    X, y = make_data(8000, seed=SEED + 21)
    X_tr, y_tr = X[:4000], y[:4000]
    X_te, y_te = X[4000:6000], y[4000:6000]
    X_bg, _ = make_data(N_BACKGROUND, seed=SEED + 1)
    p_true = 1.0 / (1.0 + np.exp(-true_logit(X_te)))
    oracle = metrics(y_te, p_true)
    print("oracle (true generating probability, reference only):", oracle)
    out = {"oracle": oracle}
    for name in LIBS:
        def run(name=name):
            row = {}
            for mono in (False, True):
                key = "mono" if mono else "free"
                raw = build(name, mono=mono).fit(X_tr, y_tr)
                row[f"{key}_raw"] = metrics(y_te, raw.predict_proba(X_te)[:, 1])
                for method in ("sigmoid", "isotonic"):
                    cal = CalibratedClassifierCV(estimator=build(name, mono=mono), method=method, cv=3)
                    cal.fit(X_tr, y_tr)
                    p = cal.predict_proba(X_te)[:, 1]
                    row[f"{key}_{method}"] = metrics(y_te, p)
                    if mono:
                        row[f"{key}_{method}_x0_violations"] = monotone_violations(cal, X_bg)[0]
            return row
        out[name] = attempt(run)
        print("calibration", name, json.dumps(out[name]), flush=True)
    RESULTS["calibration"] = out


if __name__ == "__main__":
    print("versions:", {"python": __import__("sys").version.split()[0], "xgboost": xgb.__version__,
                        "lightgbm": lgb.__version__, "sklearn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__})
    monotone_section()
    Dtr, ytr, Dte, yte = feature_section()
    cat_monotone_section(Dtr, ytr)
    calibration_section()
    print("\nRESULTS_JSON=" + json.dumps(RESULTS, default=str, indent=1))
