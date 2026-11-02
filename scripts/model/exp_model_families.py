"""Model-family comparison on the cached 2026 snapshot table: is gradient boosting really the right choice?

Same rows, same features (F6 + the 46 loadout features, no map input), same held-out matches (random match split, seed 0); training uses every
3rd second so a random forest and an MLP are feasible on CPU (evaluation uses every second). Models:
  xgb_gbdt      XGBoost boosted trees, depth 4, eta 0.05, early stopping on validation matches (the exported model's settings), GPU
  xgb_rf        XGBoost random-forest mode (many parallel trees, row and column subsampling, no boosting), GPU
  sk_rf         scikit-learn RandomForest (min_samples_leaf 100, max_features 0.4, 150 trees, 50% row subsample per tree)
  sk_rf_iso     the same forest with isotonic recalibration fitted on validation matches
  sk_et         scikit-learn ExtraTrees, same settings
  sk_mlp        scikit-learn MLP 256-128-64, Adam, standardised inputs, early stopping on the validation matches (manual epoch loop)
  ridge_logit   ridge logistic regression reference
Each: test log loss with a match-cluster interval, AUC, ECE, paired log-loss difference to xgb_gbdt, fit and predict time.

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\exp_model_families.py [--models xgb_gbdt,xgb_rf,sk_rf,sk_et,sk_mlp,ridge_logit] [--jobs 6]
Writes data/<store>/exp_model_families.json after every model.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import copy
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--models", default="xgb_gbdt,xgb_rf,sk_rf,sk_et,sk_mlp,ridge_logit")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out_path = os.path.join(W.ROOT, "data", args.store, "exp_model_families.json")
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    G = pl.read_parquet(os.path.join(W.ROOT, "data", args.store, "cache_snapshot_2026.parquet"))
    sp = W.make_split("random", G, seed=0)
    feats = W.F6 + [c for c in G.columns if c.startswith("st_")]
    sub_tr, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=args.stride)
    sub_va, Xva, yva = W.arrays(G, sp["val"], feats, stride=args.stride)
    te_sub, Xte, yte = W.arrays(G, sp["test"], feats)
    print(f"features {len(feats)}; train rows {len(ytr):,} ({len(sp['train'])} matches), val {len(yva):,}, test {len(yte):,} ({len(sp['test'])} matches)", flush=True)
    res["setup"] = {"features": len(feats), "train_rows": int(len(ytr)), "val_rows": int(len(yva)), "test_rows": int(len(yte)), "test_matches": len(sp["test"]), "stride": args.stride}
    preds = {}
    p_ref = None

    def record(name, p, fit_s, pred_s, extra=None):
        nonlocal p_ref
        if name == "xgb_gbdt":
            p_ref = p
        r = W.evaluate(te_sub, yte, p, ref=p_ref if name != "xgb_gbdt" and p_ref is not None else None)
        entry = {"logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"], "ece10": r["ece10"], "fit_seconds": fit_s, "predict_seconds": pred_s, **(extra or {})}
        if "delta_vs_ref" in r:
            entry["delta_vs_xgb_gbdt"] = r["delta_vs_ref"]
            entry["delta_vs_xgb_gbdt_ci95"] = r["delta_vs_ref_ci95"]
        res[name] = entry
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1)
        d = f"  delta vs gbdt {entry['delta_vs_xgb_gbdt']:+.4f}" if "delta_vs_xgb_gbdt" in entry else ""
        print(f"{name:<12} LL {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ECE {r['ece10']:.4f}{d}  fit {fit_s:.0f}s predict {pred_s:.0f}s", flush=True)

    want = args.models.split(",")
    if "xgb_gbdt" in want or "xgb_rf" in want or "sk_rf_iso" in want:
        pass
    if "xgb_gbdt" in want:
        t = time.time()
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda")
        fit_s = time.time() - t
        t = time.time()
        p = W.predict(m, best, Xte)
        record("xgb_gbdt", p, fit_s, time.time() - t, {"best_iteration": best})
    if "xgb_rf" in want:
        import xgboost as xgb
        t = time.time()
        m = xgb.XGBRFClassifier(n_estimators=300, max_depth=14, subsample=0.6, colsample_bynode=0.5, min_child_weight=50, learning_rate=1.0,
                                tree_method="hist", device="cuda", random_state=0)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        fit_s = time.time() - t
        t = time.time()
        p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        record("xgb_rf", p, fit_s, time.time() - t)
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    for name, cls in (("sk_rf", RandomForestClassifier), ("sk_et", ExtraTreesClassifier)):
        if name not in want:
            continue
        t = time.time()
        m = cls(n_estimators=150, min_samples_leaf=100, max_features=0.4, max_samples=0.5, bootstrap=True, n_jobs=args.jobs, random_state=0)
        m.fit(Xtr, ytr)
        fit_s = time.time() - t
        t = time.time()
        p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        pred_s = time.time() - t
        record(name, p, fit_s, pred_s)
        if name == "sk_rf":
            from sklearn.isotonic import IsotonicRegression
            pv = m.predict_proba(Xva)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(pv, yva)
            record("sk_rf_iso", np.clip(iso.predict(p), 1e-6, 1 - 1e-6), fit_s, pred_s)
        del m
    if "sk_mlp" in want:
        from sklearn.neural_network import MLPClassifier
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Ztr, Zva, Zte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        m = MLPClassifier(hidden_layer_sizes=(256, 128, 64), solver="adam", batch_size=2048, learning_rate_init=1e-3, alpha=1e-5, max_iter=1, warm_start=True, random_state=0)
        t = time.time()
        best, best_state, bad = 9.0, None, 0
        for ep in range(60):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Ztr, ytr)
            pv = np.clip(m.predict_proba(Zva)[:, 1], 1e-6, 1 - 1e-6)
            v = float(-np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv)))
            if v < best - 1e-5:
                best, best_state, bad = v, (copy.deepcopy(m.coefs_), copy.deepcopy(m.intercepts_)), 0
            else:
                bad += 1
            print(f"  mlp epoch {ep + 1}: val LL {v:.4f} (best {best:.4f})", flush=True)
            if bad >= 6:
                break
        m.coefs_, m.intercepts_ = best_state
        fit_s = time.time() - t
        t = time.time()
        p = np.clip(m.predict_proba(Zte)[:, 1], 1e-6, 1 - 1e-6)
        record("sk_mlp", p, fit_s, time.time() - t, {"epochs": ep + 1, "best_val_logloss": best})
    if "ridge_logit" in want:
        t = time.time()
        m = W.RidgeLogit().fit(Xtr, ytr)
        fit_s = time.time() - t
        t = time.time()
        p = np.clip(m.predict(Xte), 1e-6, 1 - 1e-6)
        record("ridge_logit", p, fit_s, time.time() - t)
    print("-> ", out_path)


if __name__ == "__main__":
    main()
