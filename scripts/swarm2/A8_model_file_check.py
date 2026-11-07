"""A8 check 1: saved XGBoost model file, tree count, iteration range, monotone constraints.

Read-only with respect to the project: loads models/xgb_primary_seed0.json, rebuilds the
state table with xgb_primary_seed0.build_data (same code path as the reproduction run),
and reports:
  - stored tree count, best_iteration, best_score attributes, monotone_constraints in the file
  - test log loss for each prediction range: (0,261), (0,262), (0,362) and the default
    Booster.predict call with no iteration_range
  - monotonicity of the saved model along each constrained feature (grid sweep)
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_model_file_check.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "xgb_primary_seed0.json"
META = ROOT / "models" / "xgb_primary_seed0_meta.json"
EPS = 1e-15


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    print("== A8 check 1: saved model file ==")
    print(f"model: {MODEL}  exists={MODEL.exists()}  bytes={MODEL.stat().st_size}")
    meta = json.loads(META.read_text(encoding="utf-8"))
    print(f"meta best_iteration={meta['best_iteration']} predict_iteration_range={meta['predict_iteration_range']}")

    raw = json.loads(MODEL.read_text(encoding="utf-8"))
    learner = raw["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    gbp = learner["gradient_booster"]["model"].get("gbtree_model_param", {})
    print(f"JSON: len(trees)={len(trees)}; gbtree_model_param={gbp}")
    print(f"JSON learner attributes: {learner.get('attributes', {})}")
    mc = learner.get("learner_train_param", {})
    print(f"JSON learner_train_param: {mc}")
    upd = learner["gradient_booster"]["model"]
    print(f"JSON gradient_booster keys: {list(learner['gradient_booster'].keys())}")
    print(f"JSON objective: {learner.get('objective', {})}")

    booster = xgb.Booster()
    booster.load_model(str(MODEL))
    print(f"Booster.num_boosted_rounds()={booster.num_boosted_rounds()}")
    attrs = booster.attributes()
    print(f"Booster.attributes(): {attrs}")
    cfg = json.loads(booster.save_config())
    print(f"monotone_constraints in config: {cfg['learner']['learner_train_param'].get('monotone_constraints', 'n/a')}")
    print(f"monotone constraints in config (updater): "
          f"{cfg['learner']['gradient_booster'].get('updater', 'n/a')}")

    # Rebuild the state table with the same code path as the reproduction run.
    xgbs = load("xgb_primary_seed0", ROOT / "scripts" / "model" / "xgb_primary_seed0.py")
    bo = xgbs.bo
    D = xgbs.build_data(dedupe=True)
    S, sets = D["S"], D["sets"]
    print(f"\nrebuilt states: {S.height}; kept rounds {D['n_rounds']}; kept matches {len(D['kept_final'])}")
    te = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sets["val"])))
    Xte, yte = bo.xy(te)
    Xva, yva = bo.xy(va)
    print(f"test states {len(yte)} (matches {len(sets['test'])}); val states {len(yva)} (matches {len(sets['val'])})")

    dte = xgb.DMatrix(Xte)
    ranges = {
        "iteration_range=(0,261) [261 trees, drops the best round]": (0, 261),
        "iteration_range=(0,262) [262 trees = best_iteration+1]": (0, 262),
        "iteration_range=(0,362) [all stored trees]": (0, 362),
    }
    print("\n-- test log loss by prediction range (reported 0.43660) --")
    preds = {}
    for label, rng in ranges.items():
        p = booster.predict(dte, iteration_range=rng)
        preds[label] = p
        print(f"  {label:<62} LL={ll(yte, p):.6f} Brier={float(np.mean((p - yte) ** 2)):.6f}")
    p_default = booster.predict(dte)
    print(f"  {'Booster.predict with no iteration_range (default)':<62} LL={ll(yte, p_default):.6f} "
          f"Brier={float(np.mean((p_default - yte) ** 2)):.6f}")
    p262 = preds["iteration_range=(0,262) [262 trees = best_iteration+1]"]
    print(f"max |p(0,262) - p(default)| on test: {float(np.max(np.abs(p262 - p_default))):.6f}")
    print(f"max |p(0,262) - p(0,362)| on test: {float(np.max(np.abs(p262 - preds['iteration_range=(0,362) [all stored trees]']))):.6f}")

    # Validation log loss at the same ranges (validation was the early-stopping set).
    dva = xgb.DMatrix(Xva)
    pv262 = booster.predict(dva, iteration_range=(0, 262))
    print(f"\nvalidation LL at (0,262): {ll(yva, pv262):.6f} (reported 0.43511)")

    # Monotone sweep: vary one constrained feature on a grid, others fixed, on test rows.
    feats = bo.FEATS
    mono = bo.MONO
    rng_np = np.random.default_rng(0)
    idx = rng_np.choice(len(Xte), size=min(1500, len(Xte)), replace=False)
    base = Xte[idx]
    print("\n-- monotone sweep (saved model, prediction range (0,262)) --")
    for j, s in enumerate(mono):
        if s == 0:
            continue
        vals = np.unique(Xte[:, j])
        if len(vals) > 40:
            vals = np.quantile(Xte[:, j], np.linspace(0, 1, 40))
            vals = np.unique(vals)
        viol = 0
        worst = 0.0
        n_pairs = 0
        for v_lo, v_hi in zip(vals[:-1], vals[1:]):
            a = base.copy()
            b = base.copy()
            a[:, j] = v_lo
            b[:, j] = v_hi
            pa = booster.predict(xgb.DMatrix(a), iteration_range=(0, 262))
            pb = booster.predict(xgb.DMatrix(b), iteration_range=(0, 262))
            d = pb - pa if s == 1 else pa - pb
            viol += int(np.sum(d < -1e-9))
            worst = min(worst, float(d.min()))
            n_pairs += len(d)
        print(f"  {feats[j]:<24} sign {s:+d}: grid steps {len(vals) - 1}, row-pairs {n_pairs}, "
              f"violations {viol}, worst change {worst:+.2e}")

    # Monotone check for the same sweep under the default predict (all 362 trees), for contrast.
    print("\nReading: a violation count of 0 on every constrained feature means the constraint holds on the saved model.")


if __name__ == "__main__":
    main()
