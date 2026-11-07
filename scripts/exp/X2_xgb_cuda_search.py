"""X2: XGBoost CUDA random search (14 configs, random.Random(0)) on the per-second win-probability cache.

Selection is by validation log loss only. The selected config is scored once on test with E.report("X2", ...).
The CPU baseline config is also trained on GPU and scored on test for reference (save=False, not appended).
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # noqa: E402
import xgboost as xgb  # noqa: E402

PREDS = Path(r"G:\pureskill\csds-cs2\data\viz\pt_preds")
MONO_STR = "(1,-1,1,0,0,0,0,1,0,0)"
N_BOOST = 3000
ES_ROUNDS = 100
BASELINE = {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 20, "subsample": 0.8,
            "colsample_bytree": 1.0, "reg_lambda": 1, "monotone": True}


def log(msg):
    print(msg, flush=True)


def make_params(cfg):
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": cfg["max_depth"],
        "eta": cfg["learning_rate"],
        "min_child_weight": cfg["min_child_weight"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "lambda": cfg["reg_lambda"],
        "seed": 0,
    }
    if cfg["monotone"]:
        p["monotone_constraints"] = MONO_STR
    return p


def train_cfg(cfg, dtr, dva, tag):
    params = make_params(cfg)
    for attempt in range(4):
        try:
            t0 = time.time()
            bst = xgb.train(params, dtr, num_boost_round=N_BOOST, evals=[(dva, "val")],
                            early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
            return bst, time.time() - t0
        except Exception as e:  # xgboost raises XGBoostError on CUDA failures
            msg = str(e).lower()
            if ("memory" in msg or "oom" in msg) and attempt < 3:
                log(f"[{tag}] CUDA memory error on attempt {attempt + 1}: {str(e)[:300]}; waiting 30s and retrying")
                time.sleep(30)
                continue
            raise


def main():
    t_start = time.time()
    D = E.load()
    ytr, yva, yte = D["train"]["y"], D["val"]["y"], D["test"]["y"]
    log(f"rows train={len(ytr)} val={len(yva)} test={len(yte)} X_dtype={D['train']['X'].dtype} "
        f"n_features={D['train']['X'].shape[1]}")
    dtr = xgb.DMatrix(D["train"]["X"], label=ytr)
    dva = xgb.DMatrix(D["val"]["X"], label=yva)
    dte = xgb.DMatrix(D["test"]["X"], label=yte)

    rng = random.Random(0)
    grid = []
    for i in range(14):
        grid.append({
            "idx": i,
            "max_depth": rng.choice([4, 5, 6, 8]),
            "learning_rate": rng.choice([0.03, 0.05, 0.1]),
            "min_child_weight": rng.choice([5, 20, 50]),
            "subsample": rng.choice([0.7, 0.8, 1.0]),
            "colsample_bytree": rng.choice([0.8, 1.0]),
            "reg_lambda": rng.choice([1, 5, 20]),
            "monotone": rng.choice([True, False]),
        })

    runs = []
    for cfg in grid:
        tag = f"X2 cfg{cfg['idx']:02d}"
        log(f"[{tag}] start {json.dumps(cfg)}")
        bst, secs = train_cfg(cfg, dtr, dva, tag)
        bi = int(bst.best_iteration)
        pva = bst.predict(dva, iteration_range=(0, bi + 1)).astype(np.float64)
        vll = float(np.mean(E.ll_vec(pva, yva)))
        rec = {"idx": cfg["idx"], "cfg": cfg, "best_iter": bi, "val_logloss": vll, "secs": round(secs, 1)}
        log("RESULT " + json.dumps(rec))
        runs.append((rec, bst, pva))

    runs_sorted = sorted(runs, key=lambda r: r[0]["val_logloss"])
    log("SEARCH SUMMARY (sorted by val logloss):")
    for rec, _, _ in runs_sorted:
        log(f"  cfg{rec['idx']:02d} val_ll={rec['val_logloss']:.6f} best_iter={rec['best_iter']} secs={rec['secs']} "
            f"{json.dumps(rec['cfg'])}")

    best_rec, best_bst, best_pva = runs_sorted[0]
    log(f"BEST by validation logloss: cfg{best_rec['idx']:02d} val_ll={best_rec['val_logloss']:.6f} "
        f"best_iter={best_rec['best_iter']}")

    # Single test scoring for the selected config.
    best_pte = best_bst.predict(dte, iteration_range=(0, best_rec["best_iter"] + 1)).astype(np.float64)
    PREDS.mkdir(parents=True, exist_ok=True)
    np.save(PREDS / "X2_val.npy", best_pva.astype(np.float64))
    np.save(PREDS / "X2_test.npy", best_pte)
    log(f"saved {PREDS / 'X2_val.npy'} ({best_pva.shape}) and {PREDS / 'X2_test.npy'} ({best_pte.shape})")
    E.report("X2", D["test"], best_pte, extra={
        "config": best_rec["cfg"], "best_iter": best_rec["best_iter"],
        "val_logloss": best_rec["val_logloss"], "device": "cuda", "search": "random14_rng0_val_selected"})

    # Reference: CPU baseline config re-run on GPU (colsample=1.0 and lambda=1 are XGBoost defaults here).
    base_cfg = dict(BASELINE, idx=-1)
    base_bst, base_secs = train_cfg(base_cfg, dtr, dva, "X2 baseline")
    bbi = int(base_bst.best_iteration)
    base_pva = base_bst.predict(dva, iteration_range=(0, bbi + 1)).astype(np.float64)
    base_vll = float(np.mean(E.ll_vec(base_pva, yva)))
    base_pte = base_bst.predict(dte, iteration_range=(0, bbi + 1)).astype(np.float64)
    log(f"BASELINE gpu: best_iter={bbi} val_ll={base_vll:.6f} secs={base_secs:.1f}")
    E.report("X2_baseline_gpu", D["test"], base_pte, extra={
        "config": BASELINE, "best_iter": bbi, "val_logloss": base_vll, "device": "cuda"}, save=False)

    log(f"TOTAL seconds={time.time() - t_start:.1f}")


if __name__ == "__main__":
    main()
