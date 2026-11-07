"""M5: match-cluster bootstrap intervals for the pre-round prior validation (held-out batch).

For each design (random 80/20 match split with 5 seeds; forward train 1001-1007 -> test 1008;
reverse train 1008 -> test 1001-1007) fit: constant, round only, round + map (logistic, C=1).
Report log-loss deltas on the test set with match-cluster bootstrap 95% intervals (B = 2000,
resampling test matches):
  delta_round  = LL(round only) - LL(constant)     (negative = better than constant)
  delta_map    = LL(round + map) - LL(round only)  (negative = map helps)
  delta_both   = LL(round + map) - LL(constant)
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_validation_ci.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
_spec = importlib.util.spec_from_file_location("M5_map_batch", HERE / "M5_map_batch.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

B = 2000
RNG = np.random.default_rng(7)


def row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_preds(tr, te):
    y_tr = tr["ct_win"].to_numpy()
    out = {}
    out["constant"] = np.full(te.height, y_tr.mean())
    for use_map, lab in [(False, "round"), (True, "round+map")]:
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(M.onehot(tr, use_map), y_tr)
        out[lab] = clf.predict_proba(M.onehot(te, use_map))[:, 1]
    return out


def cluster_ci(lossA, lossB, cl, B=B):
    """Mean of (lossA - lossB) over rows, bootstrapped over clusters."""
    diff = lossA - lossB
    G = cl.max() + 1
    S = np.bincount(cl, weights=diff, minlength=G)
    N = np.bincount(cl, minlength=G).astype(float)
    est = S.sum() / N.sum()
    idx = RNG.integers(0, G, size=(B, G))
    boot = S[idx].sum(1) / N[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return est, lo, hi


def evaluate(tr, te, label, rows):
    y_te = te["ct_win"].to_numpy().astype(float)
    cl = np.unique(te["match_key"].to_numpy(), return_inverse=True)[1].astype(np.int64)
    P = fit_preds(tr, te)
    L = {k: row_loss(y_te, v) for k, v in P.items()}
    for name, a, b in [("delta_round", "round", "constant"), ("delta_map", "round+map", "round"),
                       ("delta_both", "round+map", "constant")]:
        est, lo, hi = cluster_ci(L[a], L[b], cl)
        rows.append({"design": label, "contrast": name, "delta_logloss": round(est, 6),
                     "ci95_lo": round(lo, 6), "ci95_hi": round(hi, 6),
                     "ci_excludes_zero": bool(lo > 0 or hi < 0), "test_rounds": te.height,
                     "test_matches": int(cl.max() + 1)})


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    d = M.load()
    rows = []
    keys = np.array(sorted(d["match_key"].unique().to_list()))
    mk = d["match_key"].to_numpy()
    for s in range(5):
        rng = np.random.default_rng(s)
        test_keys = set(rng.choice(keys, size=int(0.2 * len(keys)), replace=False).tolist())
        is_te = np.array([k in test_keys for k in mk])
        tr = d.filter(pl.Series(~is_te))
        te = d.filter(pl.Series(is_te))
        evaluate(tr, te, f"random_seed{s}", rows)
    b1001 = d.filter(pl.col("batch") == "b1001_1007")
    b1008 = d.filter(pl.col("batch") == "b1008")
    evaluate(b1001, b1008, "forward_train1001_test1008", rows)
    evaluate(b1008, b1001, "reverse_train1008_test1001", rows)
    out = pl.DataFrame(rows)
    print(out)
    # summary of random seeds: mean of the 5 seed estimates (not a bootstrap)
    rnd = out.filter(pl.col("design").str.starts_with("random"))
    summ = rnd.group_by("contrast").agg(pl.col("delta_logloss").mean().round(6).alias("mean_over_5_seeds"),
                                        pl.col("delta_logloss").min().round(6).alias("min_seed"),
                                        pl.col("delta_logloss").max().round(6).alias("max_seed"))
    print("\nRandom split, mean over 5 seeds:")
    print(summ)


if __name__ == "__main__":
    main()
