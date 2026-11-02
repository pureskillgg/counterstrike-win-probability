"""Remaining experiments from the planning notes: other model families, and the pistol-round split.

(1) Model families on the best measured 58-input set (59 minus heavy, plus score lead), every 3rd training second (stride 3, the same rows for
    all families), frozen split, validation for early stopping, test scored once. XGBoost is the reference (paired deltas, same seconds):
    XGBoost (depth 4, eta 0.05, mcw 20, subsample 0.8, monotone alive counts), LightGBM (same depth, leaves, min hessian, bagging, monotone),
    CatBoost (depth 4, Bernoulli subsample 0.8; monotone on CPU, unconstrained on GPU), logistic regression (standardised inputs, no monotone
    constraints, no early stopping).
(2) Pistol-round split: pistol rounds (1 and 13) and the round after pistol (2 and 14) get their own XGBoost model, trained only on the
    training seconds of those rounds (early stopping on their validation seconds). Compared with the main 58-input model (seed 0, saved) on
    the same test seconds, per group, with a match-cluster bootstrap (1,000 resamples) on the difference. Also the overall test log loss when
    the dedicated predictions replace the main model's inside those rounds.

Parts (--only, comma-separated): xgb, logistic, pistol run on the GPU in the cs2gpu environment (xgb saves its predictions for pairing).
lgb and cb (LightGBM and CatBoost) pair with the saved XGBoost predictions. LightGBM runs on CPU (the pip wheel has no CUDA build).
CatBoost's GPU mode cannot take monotone constraints, so its GPU run is unconstrained and labelled as such; its CPU run keeps them.

  cs2gpu:   ...python.exe -I -X utf8 scripts\\model\\exp_families_pistol.py --only xgb,logistic,pistol --device cuda
  families: ...families_gpu\\Scripts\\python.exe -I -X utf8 scripts\\model\\exp_families_pistol.py --only cb --device cuda
  default:  ...python.exe -I -X utf8 scripts\\model\\exp_families_pistol.py --only lgb,cb --device cpu
Writes data/combined/exp_families_pistol.json. Each part is written as soon as it finishes and re-reads the file first, so parts can run
one after another without losing each other's results.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
PISTOL = [1, 13]
AFTER = [2, 14]
HEAVY = ["st_heavy_ct", "st_heavy_t"]


def match_bootstrap(keys, d_loss, n=1000, seed=0):
    """95% interval for the mean per-second difference, resampling matches."""
    _, inv = np.unique(np.asarray(keys), return_inverse=True)
    s = np.bincount(inv, weights=np.asarray(d_loss, dtype=np.float64))
    c = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def write_result(out_path, section, key, value):
    """Set one result in the shared JSON, re-reading the file first so separate processes do not overwrite each other."""
    cur = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {"families": {}, "pistol": {}}
    cur.setdefault(section, {})[key] = value
    W.dump_json(cur, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="xgb,logistic,pistol", help="comma-separated parts: xgb, logistic, lgb, cb, pistol")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    only = set(a.only.split(","))
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    v7 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v7_59_drop_heavy_scorelead.meta.json"), encoding="utf-8"))
    feats = v7["features"]
    mono = [mono59.get(f, 0) for f in feats]
    assert feats == [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]

    out_path = os.path.join(d, "exp_families_pistol.json")

    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    print(f"parts {sorted(only)} on {a.device}; train matches {len(tr)}; ({time.time() - t0:.0f}s)", flush=True)

    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    ref_va_path = os.path.join(d, "exp_families_xgb_ref_va.npy")
    ref_te_path = os.path.join(d, "exp_families_xgb_ref_te.npy")
    # paired reference: this run's XGBoost predictions, or the saved ones from an earlier xgb run when xgb is not in this run
    ref_va = ref_te = None
    if {"lgb", "cb", "logistic"} & only and "xgb" not in only:
        ref_va, ref_te = np.load(ref_va_path), np.load(ref_te_path)

    def record(name, pva, pte, bi, device):
        """Score one family on validation and test (paired with the XGBoost reference), save its predictions, and save the result at once."""
        ll_va = W.evaluate(sub_va, yva, pva)["logloss"]
        ll_te = W.evaluate(sub_te, yte, pte)["logloss"]
        r = {"validation_logloss": ll_va, "test_logloss": ll_te, "best_iteration": bi, "device": device}
        extra = ""
        if name != "xgboost":
            ev = W.evaluate(sub_te, yte, pte, ref=ref_te)
            # validation interval: the same match-cluster bootstrap as the test deltas, on the validation seconds
            d_va = logloss(yva, pva) - logloss(yva, ref_va)
            mean_va, lo_va, hi_va = match_bootstrap(sub_va["match_key"].to_numpy(), d_va)
            r.update({"test_delta_vs_xgboost": ev["delta_vs_ref"], "test_delta_ci95": ev["delta_vs_ref_ci95"],
                      "validation_delta_vs_xgboost": mean_va, "validation_delta_ci95": [lo_va, hi_va]})
            extra = (f"; validation delta vs xgboost {mean_va:+.5f} [{lo_va:+.5f}, {hi_va:+.5f}]; "
                     f"test delta vs xgboost {r['test_delta_vs_xgboost']:+.5f} "
                     f"[{r['test_delta_ci95'][0]:+.5f}, {r['test_delta_ci95'][1]:+.5f}]")
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        np.save(os.path.join(d, f"exp_families_{slug}_va.npy"), pva)
        np.save(os.path.join(d, f"exp_families_{slug}_te.npy"), pte)
        write_result(out_path, "families", name, r)
        print(f"{name}: validation {ll_va:.6f}; test {ll_te:.6f}{extra}", flush=True)

    if {"xgb", "lgb", "cb", "logistic"} & only:
        _, Xtr3, ytr3 = W.arrays(G, tr, feats, stride=3)
        print(f"stride-3 train {len(ytr3):,} seconds; validation {len(yva):,}; test {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)
    if "xgb" in only:
        m, best = W.fit_xgb(Xtr3, ytr3, Xva, yva, mono, seed=0, device=a.device)
        pva_x, pte_x = W.predict(m, best, Xva), W.predict(m, best, Xte)
        np.save(ref_va_path, pva_x)
        np.save(ref_te_path, pte_x)
        ref_va, ref_te = pva_x, pte_x
        record("xgboost", pva_x, pte_x, int(best), a.device)
        print(f"xgboost done ({time.time() - t0:.0f}s)", flush=True)
    if "lgb" in only:
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr3, ytr3)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        params = dict(objective="binary", learning_rate=0.05, num_leaves=16, max_depth=4, min_sum_hessian_in_leaf=20,
                      bagging_fraction=0.8, bagging_freq=1, monotone_constraints=mono, verbose=-1, seed=0, num_threads=-1)
        lb = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                       callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=100)])
        record("lightgbm", lb.predict(Xva, num_iteration=lb.best_iteration), lb.predict(Xte, num_iteration=lb.best_iteration),
               int(lb.best_iteration), "cpu")
        print(f"lightgbm done ({time.time() - t0:.0f}s)", flush=True)
    if "cb" in only:
        from catboost import CatBoostClassifier
        # CatBoost's GPU mode does not support monotone constraints: the GPU run is unconstrained and labelled as such;
        # the CPU run keeps the constraints, so it is the like-for-like comparison with the XGBoost reference.
        if a.device == "cuda":
            cb_kw, cb_name = {"task_type": "GPU", "devices": "0"}, "catboost (GPU, no monotone constraints)"
        else:
            cb_kw, cb_name = {"thread_count": -1, "monotone_constraints": mono}, "catboost (CPU, monotone constraints)"
        cb = CatBoostClassifier(iterations=3000, depth=4, learning_rate=0.05, loss_function="Logloss", bootstrap_type="Bernoulli",
                                subsample=0.8, early_stopping_rounds=100, verbose=False, random_seed=0, **cb_kw)
        cb.fit(Xtr3, ytr3, eval_set=(Xva, yva))
        record(cb_name, cb.predict_proba(Xva)[:, 1], cb.predict_proba(Xte)[:, 1], int(cb.get_best_iteration() or 0), a.device)
        print(f"catboost done ({time.time() - t0:.0f}s)", flush=True)
    if "logistic" in only:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr3)
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr3), ytr3)
        record("logistic", lr.predict_proba(sc.transform(Xva))[:, 1], lr.predict_proba(sc.transform(Xte))[:, 1], None, "cpu")
        print(f"logistic done ({time.time() - t0:.0f}s)", flush=True)
    if {"xgb", "lgb", "cb", "logistic"} & only:
        del Xtr3, ytr3

    if "pistol" in only:
        sub_trf, Xtrf, ytrf = W.arrays(G, tr, feats, stride=1)
        main_b = xgb.Booster()
        main_b.load_model(os.path.join(W.ROOT, "models", "winprob_v7_59_drop_heavy_scorelead.json"))
        main_best = int(v7["best_iteration"])
        p_main_te_all = main_b.predict(xgb.DMatrix(Xte), iteration_range=(0, main_best + 1))
        rte = sub_te["round"].to_numpy()
        mte = sub_te["match_key"].to_numpy()
        p_split = p_main_te_all.copy()
        for name, rounds in [("pistol rounds (1, 13)", PISTOL), ("round after pistol (2, 14)", AFTER)]:
            trm = np.isin(sub_trf["round"].to_numpy(), rounds)
            vam = np.isin(sub_va["round"].to_numpy(), rounds)
            tem = np.isin(rte, rounds)
            mb, mbest = W.fit_xgb(Xtrf[trm], ytrf[trm], Xva[vam], yva[vam], mono, seed=0, device=a.device)
            p_ded = W.predict(mb, mbest, Xte[tem])
            p_split[tem] = p_ded
            y_g = yte[tem]
            d_loss = logloss(y_g, p_ded) - logloss(y_g, p_main_te_all[tem])
            mean_d, lo, hi = match_bootstrap(mte[tem], d_loss)
            main_mean = float(logloss(y_g, p_main_te_all[tem]).mean())
            ded_mean = float(logloss(y_g, p_ded).mean())
            write_result(out_path, "pistol", name, {"train_seconds": int(trm.sum()), "test_seconds": int(tem.sum()),
                                                    "test_matches": int(len(np.unique(mte[tem]))), "main_test_logloss": main_mean,
                                                    "dedicated_test_logloss": ded_mean, "dedicated_minus_main": mean_d, "ci95": [lo, hi],
                                                    "best_iteration": int(mbest), "device": a.device})
            print(f"{name}: train seconds {int(trm.sum()):,}; test seconds {int(tem.sum()):,} in {len(np.unique(mte[tem]))} matches; "
                  f"main {main_mean:.6f} vs dedicated {ded_mean:.6f}; dedicated minus main {mean_d:+.5f} [{lo:+.5f}, {hi:+.5f}] "
                  f"({time.time() - t0:.0f}s)", flush=True)
        overall_main = float(logloss(yte, p_main_te_all).mean())
        overall_split = float(logloss(yte, p_split).mean())
        mean_all, lo_all, hi_all = match_bootstrap(mte, logloss(yte, p_split) - logloss(yte, p_main_te_all))
        write_result(out_path, "pistol", "overall_test", {"main_test_logloss": overall_main, "with_split_test_logloss": overall_split,
                                                          "split_minus_main": mean_all, "ci95": [lo_all, hi_all]})
        print(f"overall test: main {overall_main:.6f}; with pistol split {overall_split:.6f}; difference {mean_all:+.5f} "
              f"[{lo_all:+.5f}, {hi_all:+.5f}] ({time.time() - t0:.0f}s)", flush=True)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
