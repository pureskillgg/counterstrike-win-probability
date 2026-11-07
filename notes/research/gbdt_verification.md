# GBDT verification: monotone constraints, feature support, calibration

Scope: synthetic data only (largest set 8,000 rows). Run through `pixi run python -I` in `G:/pureskill/csds-cs2`. Thread counts capped at 2. No files under `tomes/` or `E:\` were touched. Versions confirmed in the run: xgboost 3.4.2, lightgbm 4.7.0, sklearn 1.9.1, pandas 3.0.6, numpy 2.5.4, Python 3.12.15.

## Summary table

| Library | Monotone constraint works (real test) | sample_weight | Early stopping on an explicit validation split | Native missing | Categorical | Calibration notes (CalibratedClassifierCV) |
|---|---|---|---|---|---|---|
| xgboost 3.4.2 `XGBClassifier` | **YES.** No explicit parameter; passed through `**kwargs`. Saved config shows `(1,0,0)`. 0 violations over a 200 x 101 grid (n=500 and n=4000). Unconstrained control: 5,708 to 5,871 violations. | Yes (`fit` kwarg). Mean P(y=1) on test rose 0.513 to 0.775 with positives weighted 5x. | Yes: `early_stopping_rounds` in constructor plus `eval_set`. Best iteration 34 (60 rounds trained). | Yes. 20% NaN in feature 1 accepted, predictions finite, test logloss 0.685. | Yes, with `enable_categorical=True` and pandas category dtype. Test logloss 0.626 with the category vs 0.703 without. A monotone constraint on the categorical column was accepted silently (enforcement not verified). | sigmoid and isotonic both run. Constrained model: sigmoid 0 violations; isotonic 529 grid steps with drops up to 4e-8, traced to float32 round-off (see below). |
| lightgbm 4.7.0 `LGBMClassifier` | **YES.** No explicit parameter; passed through `**kwargs`. Model reports `[1,0,0]`. 0 violations (n=500 and n=4000). Unconstrained control: 4,756 to 6,060 violations. | Yes. Mean P(y=1) 0.514 to 0.787. | Yes: `eval_X`/`eval_y` plus `lgb.early_stopping` callback. Best iteration 47. Note: `eval_set` is deprecated in 4.7 (`LGBMDeprecationWarning`). | Yes. Test logloss 0.676, predictions finite. | Yes. Pandas category auto-detected. Test logloss 0.614 with the category vs 0.701 without. **A monotone constraint on a categorical column aborts the process** (Fatal, not a Python exception). A constraint on numeric features with a categorical column present: 0 violations. | sigmoid and isotonic both run. Constrained model: 0 violations for both. |
| sklearn 1.9.1 `HistGradientBoostingClassifier` | **YES.** Explicit `monotonic_cst` parameter. Model reports `[1,0,0]`. 0 violations (n=500 and n=4000). Unconstrained control: 4,931 to 6,769 violations. | Yes. Mean P(y=1) 0.516 to 0.789. | **Partial.** Early stopping uses only an internal random split (`validation_fraction=0.1`). Passing `eval_set` raises `TypeError`. Stopped at 47 iterations. | Yes. Test logloss 0.676, predictions finite. | Yes, via `categorical_features="from_dtype"` with category dtype. Test logloss 0.627 with the category vs 0.706 without. A monotone constraint on a categorical feature raises `ValueError` (clean error). | sigmoid and isotonic both run. Constrained model: 0 violations for both. |

Notes on the table:
- "Monotone works" comes from the grid check in `monotone_violations`: 200 random background rows, 101 grid points on feature 0 over [-1, 1], other features held fixed. The unconstrained model is the control and shows the check is not trivially passing.
- Thousands of violations in the controls mean the test can detect non-monotonicity. The constrained runs had zero.

## Answers to the monotone questions (Q1 to Q3)

1. **xgboost 3.4.2:** `monotone_constraints` is not an explicit parameter of `XGBClassifier.__init__`, but the signature has `**kwargs`, and the value is passed through. Verified: the saved booster config contains `(1,0,0)`, and the constrained model has zero violations.
2. **lightgbm 4.7.0:** same situation. `monotone_constraints` goes through `**kwargs`. The trained model reports `[1, 0, 0]`, and the constrained model has zero violations.
3. **sklearn 1.9.1 HistGradientBoostingClassifier:** explicit `monotonic_cst` parameter. Constrained model has zero violations.

## Calibration (Q6)

Test set: 2,000 rows. Calibrator: `CalibratedClassifierCV(cv=3)`. Base estimator trained on 4,000 rows, so each calibration fold has about 1,333 rows. Format: logloss / Brier.

Oracle reference (true generating probability, not a model): logloss 0.6612, Brier 0.2343.

| Model | raw | sigmoid (Platt) | isotonic |
|---|---|---|---|
| xgboost, unconstrained | 0.6929 / 0.2471 | 0.6713 / 0.2392 | 0.6714 / 0.2393 |
| lightgbm, unconstrained | 0.6949 / 0.2479 | 0.6711 / 0.2391 | 0.6702 / 0.2387 |
| HGB, unconstrained | 0.6948 / 0.2482 | 0.6714 / 0.2393 | 0.6709 / 0.2390 |
| xgboost, constrained on x0 | 0.6901 / 0.2452 | 0.6689 / 0.2380 (0 violations) | 0.6681 / 0.2376 (529 float32-level steps, see below) |
| lightgbm, constrained on x0 | 0.6920 / 0.2463 | 0.6688 / 0.2380 (0 violations) | 0.6685 / 0.2378 (0 violations) |
| HGB, constrained on x0 | 0.6906 / 0.2460 | 0.6688 / 0.2379 (0 violations) | 0.6675 / 0.2373 (0 violations) |

Findings:
- All three estimators work inside `CalibratedClassifierCV` with both methods, including the constrained ones.
- Both calibrators improve the raw logloss by about 0.02 here.
- Sigmoid and isotonic differ by at most 0.001 logloss. That is within the noise of one seed, which I did not test.
- The sklearn calibration guide says isotonic "will perform as well as or better than 'sigmoid' when there is enough data (greater than ~1000 samples)", and that isotonic "is more prone to overfitting, especially on small datasets". It also says sigmoid is "most effective for small sample sizes". The calibration fold here is about 1,333 rows, at the edge of that threshold.
- **Usual choice at this size:** isotonic when the calibration set is well over ~1,000 rows, otherwise Platt (sigmoid). With a few thousand rows, either is defensible, and the choice should be checked on a held-out split from real data.

### The xgboost isotonic "violations" are float32 round-off, not a constraint break

- The base xgboost model has 0 violations on the same grid.
- Its `predict_proba` returns float32. `IsotonicRegression` keeps the threshold dtype, so the thresholds are float32.
- Isolated check (`calib_dtype_check.py`): float64 thresholds give 0 down-steps. float32 thresholds give 2,901 down-steps with max drop 1.19e-7. The float32 interpolation is the source.
- The 529 steps in the calibrated run have max drop 3.97e-8, consistent with this.
- I did not test a float64 cast fix inside the pipeline. If strict monotonicity in the output is required, cast the raw scores to float64 before isotonic, or use sigmoid.

## Literature (Q5)

Searched arXiv and OpenAlex with `paper_search`. Headline results from the abstracts:

- **Grinsztajn, Oyallon, Varoquaux (2022)**, arXiv:2207.08815: 45 datasets. Tree-based models remain state-of-the-art on medium-sized data (~10K samples).
- **McElfresh et al. (2023)**, arXiv:2305.02997: 19 algorithms on 176 datasets. The NN-vs-GBDT gap is often negligible. Light GBDT tuning can matter as much as the choice of model family. GBDTs handle skewed and heavy-tailed features better.
- **Ye et al. (2024), TALENT**, arXiv:2407.00956: 300+ datasets. GBDTs remain very strong baselines. Pretrained tabular models match or surpass them on many tasks, narrowing but not eliminating the gap.
- **Erickson et al. (2025), TabArena**, arXiv:2506.16791 (v4, Nov 2025): gradient-boosted trees remain strong on practical tabular data. Deep learning catches up under larger time budgets with ensembling. Foundation models excel on smaller datasets.
- **Hollmann et al. (Nature 2025), TabPFN**, doi:10.1038/s41586-024-08328-6: outperforms prior methods on datasets up to 10,000 samples. The abstract scopes the claim to small data.

Gap: none of these abstracts gives a headline result for 100K to 1M rows. The evidence for the project's 10K to 1M range is mostly at about 10K. The benchmarks above are the best available support, not direct evidence at the upper end.

## Failures, caveats, and what was not tested

- **LightGBM categorical constraint crashes the process.** A monotone constraint on a categorical column triggers `[LightGBM] [Fatal] The output cannot be monotone with respect to categorical features`. It is not a Python exception and cannot be caught. In the main run the subprocess returned 3221226505 (0xC0000409). A direct run of the probe script returned 127 under pixi. Both are non-zero with no Python traceback. The first attempt of the main script died at this step, which is why the check runs in its own subprocess.
- **xgboost accepts a monotone constraint on a categorical column without error.** Whether it is enforced was not verified.
- **Early stopping on HGB is not on the user's split.** It uses an internal 10% split. Using an explicit validation set would need a manual `warm_start` loop, which was not run.
- **Single seed, synthetic data.** Every number above is from seed 0. Differences of about 0.001 logloss are not established as real.
- **sample_weight check** is a direction test: mean predicted probability rises when positives are upweighted. It is not an exact equivalence test.
- **Monotone constraint with NaN in the constrained feature** was not tested. Missing-value handling was tested only on an unconstrained feature.
- **The grid check covers feature 0 only**, on 200 random background rows. It does not prove monotonicity everywhere.
- Calibration sample sizes are about 1,333 per fold, near the ~1,000-row threshold where isotonic becomes reliable.

## Files

- Scripts: `scripts/research/gbdt_checks.py` (main), `scripts/research/lgb_cat_monotone_probe.py`, `scripts/research/calib_debug.py`, `scripts/research/calib_dtype_check.py`
- BibTeX: `notes/research/gbdt_verification.bib`
- Literature findings recorded in project `fa6f6e09-513d-4788-9db5-43123df4ce0f` for all 8 cited papers.
- Not cited as papers: the scikit-learn calibration user guide (https://scikit-learn.org/stable/modules/calibration.html), quoted above.

## Code and output

### scripts/research/gbdt_checks.py (main run, exit 0)

```python
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
```

Output:

```text
versions: {'python': '3.12.15', 'xgboost': '3.4.2', 'lightgbm': '4.7.0', 'sklearn': '1.9.1', 'pandas': '3.0.6', 'numpy': '2.5.4'}

========================================================================
1-3. Monotone constraint on feature 0 (+1): signatures and real-fit grid check
========================================================================
signatures: {"xgboost": {"explicit_param": false, "var_kwargs": true}, "lightgbm": {"explicit_param": false, "var_kwargs": true}, "hgb": {"explicit_param": true, "var_kwargs": false}}
n=500 constrained xgboost                    violations=0      worst_drop=0.000000 config_in_model=(1,0,0)
n=500 unconstrained xgboost                  violations=5871   worst_drop=0.568380 config_in_model=()
n=500 constrained lightgbm                   violations=0      worst_drop=0.000000 config_in_model=[1, 0, 0]
n=500 unconstrained lightgbm                 violations=4756   worst_drop=0.371130 config_in_model=[]
n=500 constrained hgb                        violations=0      worst_drop=0.000000 config_in_model=[1, 0, 0]
n=500 unconstrained hgb                      violations=4931   worst_drop=0.393962 config_in_model=None
n=4000 constrained xgboost                   violations=0      worst_drop=0.000000 config_in_model=(1,0,0)
n=4000 unconstrained xgboost                 violations=5708   worst_drop=0.400260 config_in_model=()
n=4000 constrained lightgbm                  violations=0      worst_drop=0.000000 config_in_model=[1, 0, 0]
n=4000 unconstrained lightgbm                violations=6060   worst_drop=0.243481 config_in_model=[]
n=4000 constrained hgb                       violations=0      worst_drop=0.000000 config_in_model=[1, 0, 0]
n=4000 unconstrained hgb                     violations=6769   worst_drop=0.301323 config_in_model=None

========================================================================
4. sample_weight, early stopping (explicit validation split), native missing, categorical
========================================================================
sample_weight xgboost {'mean_p_plain': 0.5125, 'mean_p_weighted': 0.7754, 'weighted_higher': True}
sample_weight lightgbm {'mean_p_plain': 0.514, 'mean_p_weighted': 0.7872, 'weighted_higher': True}
sample_weight hgb {'mean_p_plain': 0.5155, 'mean_p_weighted': 0.789, 'weighted_higher': True}
early_stopping xgboost {'best_iteration': 34, 'rounds_trained': 60}
early_stopping lightgbm {'best_iteration': 47, 'trees_in_booster': 47}
early_stopping hgb {'n_iter': 47, 'validation': 'internal random split (validation_fraction=0.1)', 'explicit_eval_set_arg': "TypeError: BaseHistGradientBoosting.fit() got an unexpected keyword argument 'eval_set'"}
native_missing xgboost {'nan_train': 583, 'nan_test': 198, 'all_predictions_finite': True, 'logloss': 0.6848}
native_missing lightgbm {'nan_train': 583, 'nan_test': 198, 'all_predictions_finite': True, 'logloss': 0.6758}
native_missing hgb {'nan_train': 583, 'nan_test': 198, 'all_predictions_finite': True, 'logloss': 0.6755}
categorical_logloss xgboost {'with_cat': 0.6256, 'without_cat': 0.7026}
categorical_logloss lightgbm {'with_cat': 0.6139, 'without_cat': 0.7005}
categorical_logloss hgb {'with_cat': 0.627, 'without_cat': 0.7055}

========================================================================
3b. Monotone +1 on x0 with a categorical column present, and a constraint placed on the categorical
========================================================================
cat+monotone xgboost {'x0_constraint': {'x0_violations_with_cat': 0, 'worst_drop': 0.0}, 'constraint_on_categorical': 'accepted'}
cat+monotone lightgbm {'x0_constraint': {'x0_violations_with_cat': 0, 'worst_drop': 0.0}, 'constraint_on_categorical': {'returncode': 3221226505, 'fatal_line': ['[LightGBM] [Fatal] The output cannot be monotone with respect to categorical features'], 'fit_returned_ok': False}}
cat+monotone hgb {'x0_constraint': {'x0_violations_with_cat': 0, 'worst_drop': 0.0}, 'constraint_on_categorical': {'error': 'ValueError: Categorical features cannot have monotonic constraints.'}}

========================================================================
6. CalibratedClassifierCV (cv=3) with each estimator: sigmoid vs isotonic
========================================================================
oracle (true generating probability, reference only): {'logloss': 0.6612, 'brier': 0.2343}
calibration xgboost {"free_raw": {"logloss": 0.6929, "brier": 0.2471}, "free_sigmoid": {"logloss": 0.6713, "brier": 0.2392}, "free_isotonic": {"logloss": 0.6714, "brier": 0.2393}, "mono_raw": {"logloss": 0.6901, "brier": 0.2452}, "mono_sigmoid": {"logloss": 0.6689, "brier": 0.238}, "mono_sigmoid_x0_violations": 0, "mono_isotonic": {"logloss": 0.6681, "brier": 0.2376}, "mono_isotonic_x0_violations": 529}
calibration lightgbm {"free_raw": {"logloss": 0.6949, "brier": 0.2479}, "free_sigmoid": {"logloss": 0.6711, "brier": 0.2391}, "free_isotonic": {"logloss": 0.6702, "brier": 0.2387}, "mono_raw": {"logloss": 0.692, "brier": 0.2463}, "mono_sigmoid": {"logloss": 0.6688, "brier": 0.238}, "mono_sigmoid_x0_violations": 0, "mono_isotonic": {"logloss": 0.6685, "brier": 0.2378}, "mono_isotonic_x0_violations": 0}
calibration hgb {"free_raw": {"logloss": 0.6948, "brier": 0.2482}, "free_sigmoid": {"logloss": 0.6714, "brier": 0.2393}, "free_isotonic": {"logloss": 0.6709, "brier": 0.239}, "mono_raw": {"logloss": 0.6906, "brier": 0.246}, "mono_sigmoid": {"logloss": 0.6688, "brier": 0.2379}, "mono_sigmoid_x0_violations": 0, "mono_isotonic": {"logloss": 0.6675, "brier": 0.2373}, "mono_isotonic_x0_violations": 0}

RESULTS_JSON={
 "signature": {
  "xgboost": {
   "explicit_param": false,
   "var_kwargs": true
  },
  "lightgbm": {
   "explicit_param": false,
   "var_kwargs": true
  },
  "hgb": {
   "explicit_param": true,
   "var_kwargs": false
  }
 },
 "monotone": {
  "n=500 constrained xgboost": {
   "violations": 0,
   "worst_drop": 0.0,
   "config_in_model": "(1,0,0)"
  },
  "n=500 unconstrained xgboost": {
   "violations": 5871,
   "worst_drop": 0.56838,
   "config_in_model": "()"
  },
  "n=500 constrained lightgbm": {
   "violations": 0,
   "worst_drop": 0.0,
   "config_in_model": "[1, 0, 0]"
  },
  "n=500 unconstrained lightgbm": {
   "violations": 4756,
   "worst_drop": 0.37113,
   "config_in_model": "[]"
  },
  "n=500 constrained hgb": {
   "violations": 0,
   "worst_drop": 0.0,
   "config_in_model": "[1, 0, 0]"
  },
  "n=500 unconstrained hgb": {
   "violations": 4931,
   "worst_drop": 0.393962,
   "config_in_model": "None"
  },
  "n=4000 constrained xgboost": {
   "violations": 0,
   "worst_drop": 0.0,
   "config_in_model": "(1,0,0)"
  },
  "n=4000 unconstrained xgboost": {
   "violations": 5708,
   "worst_drop": 0.40026,
   "config_in_model": "()"
  },
  "n=4000 constrained lightgbm": {
   "violations": 0,
   "worst_drop": 0.0,
   "config_in_model": "[1, 0, 0]"
  },
  "n=4000 unconstrained lightgbm": {
   "violations": 6060,
   "worst_drop": 0.243481,
   "config_in_model": "[]"
  },
  "n=4000 constrained hgb": {
   "violations": 0,
   "worst_drop": 0.0,
   "config_in_model": "[1, 0, 0]"
  },
  "n=4000 unconstrained hgb": {
   "violations": 6769,
   "worst_drop": 0.301323,
   "config_in_model": "None"
  }
 },
 "sample_weight": {
  "xgboost": {
   "mean_p_plain": 0.5125,
   "mean_p_weighted": 0.7754,
   "weighted_higher": true
  },
  "lightgbm": {
   "mean_p_plain": 0.514,
   "mean_p_weighted": 0.7872,
   "weighted_higher": true
  },
  "hgb": {
   "mean_p_plain": 0.5155,
   "mean_p_weighted": 0.789,
   "weighted_higher": true
  }
 },
 "early_stopping": {
  "xgboost": {
   "best_iteration": 34,
   "rounds_trained": 60
  },
  "lightgbm": {
   "best_iteration": 47,
   "trees_in_booster": 47
  },
  "hgb": {
   "n_iter": 47,
   "validation": "internal random split (validation_fraction=0.1)",
   "explicit_eval_set_arg": "TypeError: BaseHistGradientBoosting.fit() got an unexpected keyword argument 'eval_set'"
  }
 },
 "native_missing": {
  "xgboost": {
   "nan_train": 583,
   "nan_test": 198,
   "all_predictions_finite": true,
   "logloss": 0.6848
  },
  "lightgbm": {
   "nan_train": 583,
   "nan_test": 198,
   "all_predictions_finite": true,
   "logloss": 0.6758
  },
  "hgb": {
   "nan_train": 583,
   "nan_test": 198,
   "all_predictions_finite": true,
   "logloss": 0.6755
  }
 },
 "categorical": {
  "xgboost": {
   "with_cat": 0.6256,
   "without_cat": 0.7026
  },
  "lightgbm": {
   "with_cat": 0.6139,
   "without_cat": 0.7005
  },
  "hgb": {
   "with_cat": 0.627,
   "without_cat": 0.7055
  }
 },
 "cat_monotone": {
  "xgboost": {
   "x0_constraint": {
    "x0_violations_with_cat": 0,
    "worst_drop": 0.0
   },
   "constraint_on_categorical": "accepted"
  },
  "lightgbm": {
   "x0_constraint": {
    "x0_violations_with_cat": 0,
    "worst_drop": 0.0
   },
   "constraint_on_categorical": {
    "returncode": 3221226505,
    "fatal_line": [
     "[LightGBM] [Fatal] The output cannot be monotone with respect to categorical features"
    ],
    "fit_returned_ok": false
   }
  },
  "hgb": {
   "x0_constraint": {
    "x0_violations_with_cat": 0,
    "worst_drop": 0.0
   },
   "constraint_on_categorical": {
    "error": "ValueError: Categorical features cannot have monotonic constraints."
   }
  }
 },
 "calibration": {
  "oracle": {
   "logloss": 0.6612,
   "brier": 0.2343
  },
  "xgboost": {
   "free_raw": {
    "logloss": 0.6929,
    "brier": 0.2471
   },
   "free_sigmoid": {
    "logloss": 0.6713,
    "brier": 0.2392
   },
   "free_isotonic": {
    "logloss": 0.6714,
    "brier": 0.2393
   },
   "mono_raw": {
    "logloss": 0.6901,
    "brier": 0.2452
   },
   "mono_sigmoid": {
    "logloss": 0.6689,
    "brier": 0.238
   },
   "mono_sigmoid_x0_violations": 0,
   "mono_isotonic": {
    "logloss": 0.6681,
    "brier": 0.2376
   },
   "mono_isotonic_x0_violations": 529
  },
  "lightgbm": {
   "free_raw": {
    "logloss": 0.6949,
    "brier": 0.2479
   },
   "free_sigmoid": {
    "logloss": 0.6711,
    "brier": 0.2391
   },
   "free_isotonic": {
    "logloss": 0.6702,
    "brier": 0.2387
   },
   "mono_raw": {
    "logloss": 0.692,
    "brier": 0.2463
   },
   "mono_sigmoid": {
    "logloss": 0.6688,
    "brier": 0.238
   },
   "mono_sigmoid_x0_violations": 0,
   "mono_isotonic": {
    "logloss": 0.6685,
    "brier": 0.2378
   },
   "mono_isotonic_x0_violations": 0
  },
  "hgb": {
   "free_raw": {
    "logloss": 0.6948,
    "brier": 0.2482
   },
   "free_sigmoid": {
    "logloss": 0.6714,
    "brier": 0.2393
   },
   "free_isotonic": {
    "logloss": 0.6709,
    "brier": 0.239
   },
   "mono_raw": {
    "logloss": 0.6906,
    "brier": 0.246
   },
   "mono_sigmoid": {
    "logloss": 0.6688,
    "brier": 0.2379
   },
   "mono_sigmoid_x0_violations": 0,
   "mono_isotonic": {
    "logloss": 0.6675,
    "brier": 0.2373
   },
   "mono_isotonic_x0_violations": 0
  }
 }
}
```

### scripts/research/lgb_cat_monotone_probe.py (run in a subprocess by the main script; standalone run below)

```python
"""Probe LightGBM monotone_constraints with a categorical column (synthetic, n=3000).

Usage: python -I lgb_cat_monotone_probe.py <variant>
  variant "x0"  -> constraints [1, 0, 0] (x0 = +1, categorical = 0)
  variant "cat" -> constraints [0, 0, 1] (constraint placed on the categorical)
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd
import lightgbm as lgb

variant = sys.argv[1]
levels = ["a", "b", "c", "d", "e"]
r = np.random.default_rng(0)
n = 3000
x0 = r.uniform(-1, 1, n)
x1 = r.uniform(-1, 1, n)
cat = r.choice(levels, size=n)
eff = {"a": -1.5, "b": -0.5, "c": 0.0, "d": 0.5, "e": 1.5}
logit = x0 + 0.5 * x1 + np.array([eff[v] for v in cat])
y = (r.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame({"x0": x0, "x1": x1, "cat": pd.Categorical(cat, categories=levels)})
cons = {"x0": [1, 0, 0], "cat": [0, 0, 1]}[variant]
print(f"variant={variant} constraints={cons} start", flush=True)
m = lgb.LGBMClassifier(n_estimators=50, num_leaves=15, monotone_constraints=cons,
                       n_jobs=2, random_state=0, verbose=-1)
m.fit(df, y)
print(f"variant={variant} fit returned OK", flush=True)
print("monotone in model:", m.booster_.dump_model().get("monotone_constraints"), flush=True)
```

Output (standalone, variant cat):

```text
=== lgb_cat_monotone_probe.py cat ===
variant=cat constraints=[0, 0, 1] start
[LightGBM] [Fatal] The output cannot be monotone with respect to categorical features
exit=127
```

### scripts/research/calib_debug.py

```python
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
```

Output:

```text
=== calib_debug.py ===
calibrated isotonic (ensemble, full)       violations=529    max_drop=3.974e-08 rows_with_violation=190
fold 0 base xgb predict_proba              violations=0      max_drop=0.000e+00 rows_with_violation=0
fold 0 calibrator monotone in raw score: False
fold 0 calibrator type: IsotonicRegression
fold 1 base xgb predict_proba              violations=0      max_drop=0.000e+00 rows_with_violation=0
fold 1 calibrator monotone in raw score: False
fold 1 calibrator type: IsotonicRegression
fold 2 base xgb predict_proba              violations=0      max_drop=0.000e+00 rows_with_violation=0
fold 2 calibrator monotone in raw score: False
fold 2 calibrator type: IsotonicRegression
recheck calibrated                         violations=529    max_drop=3.974e-08 rows_with_violation=190
example row 0 worst step index 43 p[j] 0.5663986802101135 p[j+1] 0.5663986603418986
  fold 0 base raw at j, j+1: [0.54495734 0.55245614]
  fold 1 base raw at j, j+1: [0.6739792 0.6739792]
  fold 2 base raw at j, j+1: [0.5606368 0.5606368]
exit=0
```

### scripts/research/calib_dtype_check.py

```python
"""Check whether isotonic non-monotonicity comes from float32 inputs. Synthetic data."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

r = np.random.default_rng(0)
n = 4000
s64 = np.sort(r.uniform(0.0, 1.0, n))            # float64 raw scores
y = (r.uniform(size=n) < 1 / (1 + np.exp(-(4 * s64 - 2)))).astype(float)
grid = np.linspace(0.0, 1.0, 20001)

for label, dtype in (("float64 scores", np.float64), ("float32 scores", np.float32)):
    iso = IsotonicRegression(out_of_bounds="clip").fit(s64.astype(dtype), y)
    out = iso.predict(grid)
    d = np.diff(out)
    print(f"{label:<16} dtype(X_thr)={iso.X_thresholds_.dtype} dtype(y_thr)={iso.y_thresholds_.dtype} "
          f"steps_down={int(np.sum(d < -1e-12))} max_drop={float(max(0.0, -d.min())):.3e}", flush=True)
```

Output:

```text
=== calib_dtype_check.py ===
float64 scores   dtype(X_thr)=float64 dtype(y_thr)=float64 steps_down=0 max_drop=0.000e+00
float32 scores   dtype(X_thr)=float32 dtype(y_thr)=float32 steps_down=2901 max_drop=1.192e-07
exit=0
```
