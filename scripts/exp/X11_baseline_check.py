"""X11: independent re-run of the CPU XGBoost baseline on the per-second win-probability npz cache.

Seeds 0, 1, 2. Scoring goes through pt_eval.report (match-cluster bootstrap, paired vs stored p_base).
Seed 0 is saved as X11 (reported to pt_results.jsonl); seeds 1 and 2 are saved as X11_s1 / X11_s2 and not appended.
"""
import os
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")

import numpy as np  # noqa: E402
import pt_eval as E  # noqa: E402
import xgboost as xgb  # noqa: E402

print("xgboost", xgb.__version__, "numpy", np.__version__, flush=True)
D = E.load()
Xtr, ytr = D["train"]["X"], D["train"]["y"]
Xva, yva = D["val"]["X"], D["val"]["y"]
Xte, yte = D["test"]["X"], D["test"]["y"]
print("shapes train", Xtr.shape, "val", Xva.shape, "test", Xte.shape, flush=True)
print("test p_base logloss (stored baseline):", float(E.ll_vec(D["test"]["p_base"], yte).mean()), flush=True)

MONO = (1, -1, 1, 0, 0, 0, 0, 1, 0, 0)
PRED = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
os.makedirs(PRED, exist_ok=True)

for seed in (0, 1, 2):
    name = "X11" if seed == 0 else f"X11_s{seed}"
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        tree_method="hist",
        device="cpu",
        n_jobs=4,
        monotone_constraints=MONO,
        early_stopping_rounds=100,
        eval_metric="logloss",
        random_state=seed,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best = int(m.best_iteration)
    fit_sec = time.time() - t0
    pv = np.asarray(m.predict_proba(Xva, iteration_range=(0, best + 1))[:, 1], dtype=np.float64)
    pt = np.asarray(m.predict_proba(Xte, iteration_range=(0, best + 1))[:, 1], dtype=np.float64)
    np.save(os.path.join(PRED, f"{name}_val.npy"), pv)
    np.save(os.path.join(PRED, f"{name}_test.npy"), pt)
    val_ll = float(E.ll_vec(pv, yva).mean())
    print(f"seed={seed} best_iter={best} fit_sec={fit_sec:.1f} val_logloss={val_ll:.5f}", flush=True)
    rep = E.report(
        name,
        D["test"],
        pt,
        extra={"seed": seed, "best_iter": best, "fit_seconds": round(fit_sec, 1), "device": "cpu", "n_jobs": 4,
               "script": "scripts/exp/X11_baseline_check.py"},
        save=(seed == 0),
    )
    print(f"SUMMARY seed={seed} test_logloss={rep['logloss']:.5f} ci95={rep['logloss_ci95']} "
          f"rw={rep['logloss_round_weighted']:.5f} brier={rep['brier']:.5f} auc={rep['auc']:.5f} "
          f"ece={rep['ece10']:.5f} delta_vs_base={rep['vs_cpu_xgb_baseline']:.5f} "
          f"delta_ci95={rep['vs_cpu_xgb_baseline_ci95']}", flush=True)

print("DONE", flush=True)
