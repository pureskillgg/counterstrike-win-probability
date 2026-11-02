"""Is XGBoost still the right boosted-tree library here? XGBoost vs LightGBM vs CatBoost on identical rows, inputs and held-out matches.

  G:\\pureskill\\envs\\tabicl\\Scripts\\python.exe -X utf8 scripts\\model\\exp_gbdt_compare.py [--device-xgb cuda]

Same table, 52 inputs, random match split (seed 0), monotone constraints on the alive counts (+ for alive_ct and ct_alive_minus_t, - for alive_t) and
early stopping on the validation matches as exp_model_families.py; training uses every third second, scoring uses every second of the held-out matches.
Each model is scored with the project's match-cluster bootstrap and compared PAIRED with the XGBoost reference. Writes data/<store>/exp_gbdt_compare.json
after every model. Reported: log loss, AUC, ECE (10 bins, equal count 20, smooth), fit time, predict time per 100,000 rows, and best iteration.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device-xgb", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--models", default="xgb,lgbm,cat4,cat6")
    a = ap.parse_args()
    out_path = os.path.join(W.ROOT, "data", a.store, "exp_gbdt_compare.json")
    res = {}
    G = pl.read_parquet(os.path.join(W.ROOT, "data", a.store, "cache_snapshot_2026.parquet"))
    sp = W.make_split("random", G, seed=0)
    feats = W.F6 + [c for c in G.columns if c.startswith("st_")]
    mono = W.mono_for(feats)
    _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=a.stride)
    _, Xva, yva = W.arrays(G, sp["val"], feats, stride=a.stride)
    te_sub, Xte, yte = W.arrays(G, sp["test"], feats)
    print(f"train {len(ytr):,} val {len(yva):,} test {len(yte):,} rows; {len(feats)} inputs; monotone on {sum(1 for m in mono if m)}", flush=True)
    preds = {}

    def record(name, p, fit_s, pred_s, best, extra=None):
        r = W.evaluate(te_sub, yte, p, ref=preds.get("xgb"))
        e = {"logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"], "ece20_equal_count": r["ece20_equal_count"], "smooth_ece": r["smooth_ece"],
             "fit_seconds": fit_s, "predict_seconds_per_100k_rows": pred_s / len(yte) * 1e5, "best_iteration": best, **(extra or {})}
        if "delta_vs_ref" in r:
            e["delta_vs_xgb"], e["delta_ci95"] = r["delta_vs_ref"], r["delta_vs_ref_ci95"]
        res[name] = e
        W.dump_json(res, out_path)
        d = f" delta vs xgb {e['delta_vs_xgb']:+.4f} {[round(x, 4) for x in e['delta_ci95']]}" if "delta_vs_xgb" in e else ""
        print(f"{name:<6} LL {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ECE {r['ece10']:.4f}{d} | fit {fit_s:.0f}s, predict {e['predict_seconds_per_100k_rows']:.2f}s/100k rows, it {best}", flush=True)

    want = a.models.split(",")
    if "xgb" in want:
        t = time.time()
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device=a.device_xgb)
        fit_s = time.time() - t
        t = time.time()
        p = W.predict(m, best, Xte)
        preds["xgb"] = p
        record("xgb", p, fit_s, time.time() - t, best, {"device": a.device_xgb})
    if "lgbm" in want:
        import lightgbm as lgb
        t = time.time()
        m = lgb.LGBMClassifier(objective="binary", n_estimators=3000, learning_rate=0.05, num_leaves=15, max_depth=4, min_child_samples=20, subsample=0.8, subsample_freq=1,
                               monotone_constraints=list(mono), n_jobs=a.threads, random_state=0, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
        fit_s = time.time() - t
        t = time.time()
        p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        record("lgbm", p, fit_s, time.time() - t, int(m.best_iteration_), {"device": "cpu"})
    for name, depth in (("cat4", 4), ("cat6", 6)):
        if name not in want:
            continue
        from catboost import CatBoostClassifier
        t = time.time()
        m = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=depth, loss_function="Logloss", monotone_constraints=[int(x) for x in mono], thread_count=a.threads,
                               random_seed=0, early_stopping_rounds=100, verbose=0, allow_writing_files=False)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        fit_s = time.time() - t
        t = time.time()
        p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        record(name, p, fit_s, time.time() - t, int(m.get_best_iteration()), {"device": "cpu"})
    print("->", out_path)


if __name__ == "__main__":
    main()
