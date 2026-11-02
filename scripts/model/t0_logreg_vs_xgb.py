"""T0 (Meridian pin 90519962): does the tuned XGBoost beat a logistic regression on the same 57 inputs by at least 0.005 validation log loss?

Arms, all-maps population of scripts/model/sitecheck_xgb.py (same train and validation rows):
  XGBoost arm A: the saved validation predictions of sitecheck_xgb.py, seeds 0, 1, 2 (data/diagnostics/preds/allmaps_xgb_A_seed<s>.parquet).
  Logistic regression: scikit-learn defaults (L2 penalty, C = 1.0, lbfgs, max_iter = 100) on the 57 inputs, standardised with
  train-row mean and standard deviation, fitted on the training seconds.

Rule (pin 90519962): pass if logistic minus XGBoost validation log loss is at least +0.005 with a match-clustered 95% CI lower bound above 0.
Reading recorded here: XGBoost is the mean of the three seeds' per-second losses, so the decision uses one difference per second. The per-seed
differences are reported beside it. Match-cluster bootstrap, 1,000 resamples, seed 0.

Writes data/diagnostics/t0_logreg_vs_xgb.json. Run in the cs2gpu environment (polars, scikit-learn):
  C:/Users/<user>/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/t0_logreg_vs_xgb.py
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
PREDS = os.path.join(DIAG, "preds")
OUT = os.path.join(DIAG, "t0_logreg_vs_xgb.json")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
SEEDS = [0, 1, 2]
N_BOOT = 1000


def bootstrap_mean(match, d, seed=0):
    """Mean of d over seconds; match-cluster bootstrap with N_BOOT resamples."""
    uniq, inv = np.unique(match, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=uniq.size)
    cnt = np.bincount(inv, minlength=uniq.size).astype(float)
    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, uniq.size, uniq.size)
        boot[b] = sums[pick].sum() / cnt[pick].sum()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)], "rows": int(d.size), "matches": int(uniq.size)}


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    geo_old = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    geo_new = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null()).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    # the same population as sitecheck_xgb.py, built on the narrow key columns first; the feature columns are gathered once
    pop = (C.select(KEY + ["label_ct_win"])
             .join(geo_old.select(KEY).unique(KEY), on=KEY, how="semi")
             .join(geo_new.select(KEY).unique(KEY), on=KEY, how="semi")
             .join(A, on="match_key", how="inner"))
    pop = pop.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    G = pop.join(C.select(KEY + feats), on=KEY, how="left", maintain_order="left")
    del C, pop, geo_old, geo_new
    is_tr = (G["split"] == "train").to_numpy()
    is_va = (G["split"] == "val").to_numpy()
    n_tr, n_va = int(is_tr.sum()), int(is_va.sum())
    print(f"train rows {n_tr:,}, validation rows {n_va:,} ({time.time() - t0:.0f} s)", flush=True)

    # float64 matrices filled one column at a time (no 57-column temporary)
    Xtr = np.empty((n_tr, len(feats)), dtype=np.float64)
    Xva = np.empty((n_va, len(feats)), dtype=np.float64)
    for j, c in enumerate(feats):
        col = G[c].to_numpy().astype(np.float64)
        Xtr[:, j] = col[is_tr]
        Xva[:, j] = col[is_va]
        del col
    ytr = G["label_ct_win"].to_numpy()[is_tr].astype(np.int64)
    base = (G.filter(pl.col("split") == "val").select(KEY + ["label_ct_win"])
              .rename({"label_ct_win": "label"}).with_columns(pl.col("label").cast(pl.Int64)))
    del G
    # standardise column by column with the training statistics (population std, as np.std)
    mu = np.empty(len(feats))
    sd = np.empty(len(feats))
    for j in range(len(feats)):
        mu[j] = Xtr[:, j].mean()
        sd[j] = Xtr[:, j].std()
    sd[sd == 0] = 1.0
    for j in range(len(feats)):
        Xtr[:, j] = (Xtr[:, j] - mu[j]) / sd[j]
        Xva[:, j] = (Xva[:, j] - mu[j]) / sd[j]
    print(f"standardised ({time.time() - t0:.0f} s)", flush=True)

    clf = LogisticRegression()  # scikit-learn defaults, as pinned
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        clf.fit(Xtr, ytr)
    conv_warn = any("ConvergenceWarning" in str(w.category) for w in wlist)
    print(f"logistic fitted: n_iter {int(np.max(clf.n_iter_))}, convergence warning {conv_warn} ({time.time() - t0:.0f} s)", flush=True)
    base = base.with_columns(pl.Series("p_log", clf.predict_proba(Xva)[:, 1]))
    n_train = int(ytr.size)
    del Xtr, Xva, ytr

    match = base["match_key"].to_numpy()
    y = base["label"].to_numpy().astype(np.int64)
    p_log = base["p_log"].to_numpy()
    L_log = W.ll_vec(p_log, y)
    per_seed, L_xgb = {}, []
    for s in SEEDS:
        P = (pl.read_parquet(os.path.join(PREDS, f"allmaps_xgb_A_seed{s}.parquet"))
               .filter(pl.col("split") == "val").select(KEY + ["label", "p"]).sort(KEY)
               .rename({"label": "label_xgb"}))
        J = base.join(P, on=KEY, how="inner", maintain_order="left")
        assert J.height == base.height, f"seed {s}: {J.height:,} matched rows, expected {base.height:,}"
        assert (J["label_xgb"].to_numpy().astype(np.int64) == y).all(), f"seed {s}: labels do not align"
        p_x = J["p"].to_numpy()
        L_xgb.append(W.ll_vec(p_x, y))
        per_seed[s] = {"logreg_minus_xgb_A": paired(match, y, p_log, p_x),
                       "xgb_A_validation_log_loss": float(L_xgb[-1].mean())}
        print(f"seed {s}: xgb A {per_seed[s]['xgb_A_validation_log_loss']:.6f}, "
              f"logreg minus xgb A {per_seed[s]['logreg_minus_xgb_A']['diff']:+.6f} "
              f"CI {per_seed[s]['logreg_minus_xgb_A']['ci95']}", flush=True)
    d = L_log - np.mean(L_xgb, axis=0)
    dec = bootstrap_mean(match, d)
    passes = bool(dec["diff"] >= 0.005 and dec["ci95"][0] > 0)
    res = {"decision_set": "validation seconds", "rule": "pass if logistic minus XGBoost validation log loss >= +0.005 with a match-clustered 95% CI lower bound above 0 (XGBoost = mean over seeds 0, 1, 2)",
           "logreg_validation_log_loss": float(L_log.mean()), "xgb_mean_over_seeds_validation_log_loss": float(np.mean(L_xgb)),
           "logreg_minus_xgb_mean_over_seeds": dec, "passes": passes,
           "logreg": {"n_iter": int(np.max(clf.n_iter_)), "convergence_warning": conv_warn, "train_rows": n_train, "validation_rows": int(y.size)},
           "per_seed": per_seed, "seconds": round(time.time() - t0)}
    W.dump_json(res, OUT)
    print(json.dumps({"logreg_minus_xgb_mean_over_seeds": dec, "passes": passes}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
