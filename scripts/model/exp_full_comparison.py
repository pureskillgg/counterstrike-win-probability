"""Model families on the full frozen split: every second of every training match, the same 59 inputs as winprob_v3, and the same test matches
for every model.   (XGBoost on the GPU when available; the rest on the CPU)

  ...python.exe -I -X utf8 scripts\\model\\exp_full_comparison.py [--models xgb,lgbm,catboost,logit,hgb,rf,et,mlp] [--device cuda] [--xgb-model winprob_v4]
  (--xgb-model loads a model already fitted on the GPU and saved in models/, with its best iteration, instead of fitting XGBoost again.)

Training = every second of the frozen training matches (stride 1, no thinning), validation = every second of the validation matches (early
stopping where the family supports it), test = every second of the frozen test matches. Random forest, ExtraTrees and the MLP are too slow to
fit on all rows in one run, so they fit a random sample of training rows (--sub-rf, --sub-mlp); the sample fraction is written to the output.
Every model is scored with wp_common.evaluate on the same test rows. The headline is per-second log loss; round-weighted log loss is reported
next to it, and each model's rank under both is written so a change of ranking is visible.
Writes data/<store>/exp_full_comparison.json.
"""
import argparse
import json
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--models", default="xgb,lgbm,catboost,logit,hgb,rf,et,mlp")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="device for xgb only (the other families run on the CPU)")
    ap.add_argument("--sub-rf", type=float, default=0.1, help="fraction of training rows each random forest / ExtraTrees fit uses")
    ap.add_argument("--sub-mlp", type=float, default=0.25, help="fraction of training rows the MLP fits on")
    ap.add_argument("--xgb-model", default=None, help="name in models/ of an XGBoost model already fitted (GPU); loaded instead of refitted")
    ap.add_argument("--out", default="exp_full_comparison.json", help="output file name in data/<store>/ (run parts separately, then merge_comparison.py)")
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    names = [n for n in a.models.split(",") if n]
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    # Streaming collect with float32 features: polars casts in batches, so the 2026 rows never exist as a full float64 frame
    # (that frame was about 11 GB and took the WSL VM down during the load). W.arrays converts to float32 anyway, so the values
    # are the same.
    G = lf.with_columns(pl.col(feats).cast(pl.Float32)).collect(engine="streaming")
    sub_tr, Xtr, ytr = W.arrays(G, sp["train"], feats)
    sub_va, Xva, yva = W.arrays(G, sp["val"], feats)
    te_sub, Xte, yte = W.arrays(G, sp["test"], feats)
    del G
    mono = list(W.mono_for(feats))
    print(f"rows: train {len(ytr):,} (every second of {len(sp['train'])} matches), val {len(yva):,} ({len(sp['val'])}), "
          f"test {len(yte):,} ({len(sp['test'])}); {len(feats)} inputs ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(0)
    out = {"inputs": len(feats), "stride": 1, "train_matches": len(sp["train"]), "val_matches": len(sp["val"]), "test_matches": len(sp["test"]),
           "train_rows": int(len(ytr)), "val_rows": int(len(yva)), "test_rows": int(len(yte)), "models": {}}
    preds = {}

    def record(name, p, fit_s, note, pva=None):
        p = np.asarray(p, dtype=np.float64)
        preds[name] = p
        r = W.evaluate(te_sub, yte, p, ref=preds["xgb"] if "xgb" in preds and name != "xgb" else None)
        row = {"logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "logloss_round_weighted": r["logloss_round_weighted"],
               "logloss_round_weighted_ci95": r["logloss_round_weighted_ci95"], "auc": r["auc"], "ece10": r["ece10"], "fit_seconds": round(fit_s),
               "note": note}
        if name != "xgb" and "delta_vs_ref" in r:
            row["delta_vs_xgb"] = r["delta_vs_ref"]
            row["delta_vs_xgb_ci95"] = r["delta_vs_ref_ci95"]
        if pva is not None:
            row["validation_logloss"] = W.evaluate(sub_va, yva, np.asarray(pva, dtype=np.float64))["logloss"]
        out["models"][name] = row
        print(f"{name}: log loss {r['logloss']:.5f} {[round(x, 5) for x in r['logloss_ci95']]}  round-weighted {r['logloss_round_weighted']:.5f}  "
              f"AUC {r['auc']:.4f}  ({fit_s:.0f}s) {note}", flush=True)

    if "xgb" in names and a.xgb_model:
        import xgboost as xgb
        t = time.time()
        meta = json.load(open(os.path.join(W.ROOT, "models", f"{a.xgb_model}.meta.json"), encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(os.path.join(W.ROOT, "models", f"{a.xgb_model}.json"))
        best = int(meta["best_iteration"])

        def pred(X):
            return booster.predict(xgb.DMatrix(X), iteration_range=(0, best + 1))

        record("xgb", pred(Xte), time.time() - t, f"{a.xgb_model}, loaded from models/: best_iteration {best}, fitted on {meta['device']}",
               pva=pred(Xva))
    elif "xgb" in names:
        t = time.time()
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device=a.device)
        record("xgb", W.predict(m, best, Xte), time.time() - t, f"v3 settings, early stop on validation, best_iteration {best}, device {a.device}",
               pva=W.predict(m, best, Xva))
    if "lgbm" in names:
        import lightgbm as lgb
        t = time.time()
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                               monotone_constraints=mono, random_state=0, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
        record("lgbm", m.predict_proba(Xte, num_iteration=m.best_iteration_)[:, 1], time.time() - t,
               f"leaves 31, lr 0.05, early stop on validation, best_iteration {m.best_iteration_}",
               pva=m.predict_proba(Xva, num_iteration=m.best_iteration_)[:, 1])
    if "catboost" in names:
        from catboost import CatBoostClassifier
        t = time.time()
        m = CatBoostClassifier(iterations=3000, learning_rate=0.08, depth=6, loss_function="Logloss", eval_metric="Logloss", random_seed=0,
                               od_type="Iter", od_wait=100, monotone_constraints=mono, use_best_model=True, verbose=0, thread_count=-1)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        record("catboost", m.predict_proba(Xte)[:, 1], time.time() - t, f"depth 6, lr 0.08, early stop on validation, best_iteration {m.get_best_iteration()}",
               pva=m.predict_proba(Xva)[:, 1])
    if "logit" in names:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        t = time.time()
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300))
        m.fit(Xtr, ytr)
        record("logit", m.predict_proba(Xte)[:, 1], time.time() - t, "standardised, C 1.0, no interactions", pva=m.predict_proba(Xva)[:, 1])
    if "hgb" in names:
        from sklearn.ensemble import HistGradientBoostingClassifier
        t = time.time()
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=50, monotonic_cst=mono,
                                           early_stopping=False, random_state=0)
        m.fit(Xtr, ytr)
        record("hgb", m.predict_proba(Xte)[:, 1], time.time() - t, "600 iterations, no early stopping (sklearn's own early stopping uses a random row split)",
               pva=m.predict_proba(Xva)[:, 1])
    if "rf" in names or "et" in names:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        sel = rng.choice(len(ytr), size=int(a.sub_rf * len(ytr)), replace=False)
        if "rf" in names:
            t = time.time()
            m = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=0)
            m.fit(Xtr[sel], ytr[sel])
            record("rf", m.predict_proba(Xte)[:, 1], time.time() - t, f"100 trees, min leaf 50, fit on {a.sub_rf:.0%} of training rows ({len(sel):,})",
                   pva=m.predict_proba(Xva)[:, 1])
        if "et" in names:
            t = time.time()
            m = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=50, n_jobs=-1, random_state=0)
            m.fit(Xtr[sel], ytr[sel])
            record("et", m.predict_proba(Xte)[:, 1], time.time() - t, f"100 trees, min leaf 50, fit on {a.sub_rf:.0%} of training rows ({len(sel):,})",
                   pva=m.predict_proba(Xva)[:, 1])
    if "mlp" in names:
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        sel = rng.choice(len(ytr), size=int(a.sub_mlp * len(ytr)), replace=False)
        t = time.time()
        m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), batch_size=4096, learning_rate_init=1e-3, max_iter=10,
                                                          random_state=0))
        m.fit(Xtr[sel], ytr[sel])
        record("mlp", m.predict_proba(Xte)[:, 1], time.time() - t, f"128-64, 10 epochs, fit on {a.sub_mlp:.0%} of training rows ({len(sel):,})",
               pva=m.predict_proba(Xva)[:, 1])

    ll = sorted(out["models"], key=lambda k: out["models"][k]["logloss"])
    rw = sorted(out["models"], key=lambda k: out["models"][k]["logloss_round_weighted"])
    for i, k in enumerate(ll):
        out["models"][k]["rank_per_second"] = i + 1
        out["models"][k]["rank_round_weighted"] = rw.index(k) + 1
    out["ranking_per_second"] = ll
    out["ranking_round_weighted"] = rw
    W.dump_json(out, os.path.join(d, a.out))
    print(f"ranking, per second: {ll}")
    print(f"ranking, round-weighted: {rw}")
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
