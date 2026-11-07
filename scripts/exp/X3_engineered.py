import os
import sys
import time
import json

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\model")
import pt_eval as E  # sets MKL_THREADING_LAYER before numpy is imported

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

PRED_DIR = r"G:\pureskill\csds-cs2\data\viz\pt_preds"
os.makedirs(PRED_DIR, exist_ok=True)

BASE_MONO = [1, -1, 1, 0, 0, 0, 0, 1, 0, 0]
NEW = ["time_left", "bomb_time_left", "alive_product", "alive_sum", "ct_share",
       "diff_x_time", "diff_x_planted", "score_x_diff", "second_half"]


def engineer(X):
    X = np.asarray(X, dtype=np.float64)
    a_ct, a_t, diff = X[:, 0], X[:, 1], X[:, 2]
    planted = X[:, 3] > 0.5
    t_live, sp, rnd, score = X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    new = np.column_stack([
        115.0 - t_live,
        np.where(planted, 40.0 - sp, -1.0),
        a_ct * a_t,
        a_ct + a_t,
        a_ct / np.maximum(a_ct + a_t, 1.0),
        diff * t_live,
        diff * X[:, 3],
        score * diff,
        ((rnd >= 13) & (rnd <= 24)).astype(np.float64),
    ])
    return np.hstack([X, new])


def fit(cols, mono, Ftr, ytr, Fva, yva, Fte, tag):
    Xtr = Ftr[:, cols]
    Xva = Fva[:, cols]
    Xte = Fte[:, cols]
    for attempt in range(4):
        try:
            t0 = time.time()
            m = XGBClassifier(
                n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                subsample=0.8, tree_method="hist", device="cuda",
                monotone_constraints=tuple(mono), early_stopping_rounds=100,
                eval_metric="logloss", random_state=0, n_jobs=4)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            bi = int(m.best_iteration)
            pv = m.predict_proba(Xva, iteration_range=(0, bi + 1))[:, 1].astype(np.float64)
            pt = m.predict_proba(Xte, iteration_range=(0, bi + 1))[:, 1].astype(np.float64)
            vll = float(E.ll_vec(pv, yva).mean())
            print(f"[{tag}] ncols={len(cols)} best_iter={bi} val_ll={vll:.6f} secs={time.time() - t0:.1f}", flush=True)
            return {"tag": tag, "bi": bi, "vll": vll, "pv": pv, "pt": pt}
        except Exception as e:
            msg = str(e)
            if attempt < 3 and ("memory" in msg.lower() or "oom" in msg.lower()):
                print(f"[{tag}] CUDA OOM on attempt {attempt + 1}: {msg[:200]}; waiting 30s", flush=True)
                time.sleep(30)
                continue
            raise


def main():
    t_start = time.time()
    D = E.load()
    Xtr_raw = D["train"]["X"]
    Ftr = engineer(Xtr_raw)
    Fva = engineer(D["val"]["X"])
    Fte = engineer(D["test"]["X"])
    ytr = np.asarray(D["train"]["y"]).astype(np.int64)
    yva = np.asarray(D["val"]["y"]).astype(np.int64)
    print("shapes", Ftr.shape, Fva.shape, Fte.shape, "xgboost", xgb.__version__, flush=True)
    print("base cols equal raw:", bool(np.allclose(Ftr[:, :10], Xtr_raw)), flush=True)
    print("planted flag agrees with seconds_since_plant>=0 (train):",
          float(np.mean((Xtr_raw[:, 3] > 0.5) == (Xtr_raw[:, 5] >= 0))), flush=True)

    base_cols = list(range(10))
    all_cols = list(range(19))
    mono_all = BASE_MONO + [0] * len(NEW)

    base = fit(base_cols, BASE_MONO, Ftr, ytr, Fva, yva, Fte, "base10")
    rep_base = E.report("X3_base10_gpu_replicate", D["test"], base["pt"], save=False)

    allf = fit(all_cols, mono_all, Ftr, ytr, Fva, yva, Fte, "all19")
    np.save(os.path.join(PRED_DIR, "X3_val.npy"), allf["pv"].astype(np.float64))
    np.save(os.path.join(PRED_DIR, "X3_test.npy"), allf["pt"].astype(np.float64))
    rep_all = E.report("X3", D["test"], allf["pt"], extra={"best_iter": allf["bi"], "val_logloss": allf["vll"],
                                                             "feature_set": "base10+9 engineered"})

    singles = {}
    for j, name in enumerate(NEW):
        cols = base_cols + [10 + j]
        singles[name] = fit(cols, BASE_MONO + [0], Ftr, ytr, Fva, yva, Fte, f"single_{name}")

    best_name = min(singles, key=lambda k: singles[k]["vll"])
    best = singles[best_name]
    np.save(os.path.join(PRED_DIR, f"X3_single_{best_name}_val.npy"), best["pv"].astype(np.float64))
    np.save(os.path.join(PRED_DIR, f"X3_single_{best_name}_test.npy"), best["pt"].astype(np.float64))
    rep_best = E.report(f"X3_single_{best_name}", D["test"], best["pt"],
                        extra={"best_iter": best["bi"], "val_logloss": best["vll"]})

    print("VAL SUMMARY (logloss, lower is better)", flush=True)
    print(f"  base10 (GPU replicate): {base['vll']:.6f} best_iter={base['bi']}", flush=True)
    print(f"  all19:                  {allf['vll']:.6f} best_iter={allf['bi']} delta={allf['vll'] - base['vll']:+.6f}", flush=True)
    for name in NEW:
        s = singles[name]
        print(f"  single {name:16s}: {s['vll']:.6f} best_iter={s['bi']} delta={s['vll'] - base['vll']:+.6f}", flush=True)
    print(f"BEST SINGLE (val): {best_name}", flush=True)

    summary = {
        "base10_val_ll": base["vll"], "all19_val_ll": allf["vll"], "best_single": best_name,
        "best_single_val_ll": best["vll"],
        "all19_best_iter": allf["bi"], "best_single_best_iter": best["bi"],
        "single_val_ll": {k: v["vll"] for k, v in singles.items()},
        "base10_replicate_test": {"logloss": rep_base["logloss"], "ci95": rep_base["logloss_ci95"],
                                  "round_weighted": rep_base["logloss_round_weighted"], "brier": rep_base["brier"],
                                  "auc": rep_base["auc"], "ece10": rep_base["ece10"],
                                  "vs_base": rep_base["vs_cpu_xgb_baseline"]},
        "all19_test": {"logloss": rep_all["logloss"], "ci95": rep_all["logloss_ci95"],
                       "round_weighted": rep_all["logloss_round_weighted"], "brier": rep_all["brier"],
                       "auc": rep_all["auc"], "ece10": rep_all["ece10"],
                       "vs_base": rep_all["vs_cpu_xgb_baseline"], "vs_base_ci95": rep_all["vs_cpu_xgb_baseline_ci95"]},
        "best_single_test": {"logloss": rep_best["logloss"], "ci95": rep_best["logloss_ci95"],
                             "round_weighted": rep_best["logloss_round_weighted"], "brier": rep_best["brier"],
                             "auc": rep_best["auc"], "ece10": rep_best["ece10"],
                             "vs_base": rep_best["vs_cpu_xgb_baseline"],
                             "vs_base_ci95": rep_best["vs_cpu_xgb_baseline_ci95"]},
        "total_secs": time.time() - t_start,
    }
    print("SUMMARY_JSON " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
