"""M5: do batch-specific map priors help within a batch (batch known), compared with pooled priors?

Within-batch random match splits (80/20, 5 seeds) on each batch alone:
  - pooled prior: map + round logistic trained on ALL clean rounds from the other 80% of matches
    (the pooled training set includes the other batch), tested on the held-out 20% of the batch.
  - batch-specific prior: same model trained only on the 80% of matches from the same batch.
  - round only (no map) trained on the same batch-specific set, for the increment.
Contrast: LL(batch-specific map model) - LL(batch-specific round only), and
          LL(pooled map model) - LL(batch-specific round only). Match-cluster bootstrap 95% intervals.

Also reports the in-batch increment of the map term (batch-specific) for each batch, so the
parent can see whether a batch-aware map prior would carry any value even when the batch is known.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M5_batch_aware.py
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
RNG = np.random.default_rng(11)


def row_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit(tr, te, use_map):
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(M.onehot(tr, use_map), tr["ct_win"].to_numpy())
    return clf.predict_proba(M.onehot(te, use_map))[:, 1]


def ci(a, b, cl):
    diff = a - b
    G = cl.max() + 1
    S = np.bincount(cl, weights=diff, minlength=G)
    N = np.bincount(cl, minlength=G).astype(float)
    est = S.sum() / N.sum()
    idx = RNG.integers(0, G, size=(B, G))
    boot = S[idx].sum(1) / N[idx].sum(1)
    return est, *np.percentile(boot, [2.5, 97.5])


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    d = M.load()
    rows = []
    for batch in ["b1008", "b1001_1007"]:
        for s in range(5):
            keys = np.array(sorted(d.filter(pl.col("batch") == batch)["match_key"].unique().to_list()))
            rng = np.random.default_rng(100 + s)
            test_keys = set(rng.choice(keys, size=int(0.2 * len(keys)), replace=False).tolist())
            mk = d["match_key"].to_numpy()
            in_b = (d["batch"] == batch).to_numpy()
            is_te = np.array([k in test_keys for k in mk]) & in_b
            is_tr_same = np.array([k not in test_keys for k in mk]) & in_b
            is_tr_all = np.array([k not in test_keys for k in mk])
            te = d.filter(pl.Series(is_te))
            tr_same = d.filter(pl.Series(is_tr_same))
            tr_all = d.filter(pl.Series(is_tr_all))
            y = te["ct_win"].to_numpy().astype(float)
            cl = np.unique(te["match_key"].to_numpy(), return_inverse=True)[1].astype(np.int64)
            p_round_same = fit(tr_same, te, False)
            p_map_same = fit(tr_same, te, True)
            p_map_pool = fit(tr_all, te, True)
            L_r = row_loss(y, p_round_same)
            L_ms = row_loss(y, p_map_same)
            L_mp = row_loss(y, p_map_pool)
            e1, l1, h1 = ci(L_ms, L_r, cl)
            e2, l2, h2 = ci(L_mp, L_r, cl)
            rows.append({"test_batch": batch, "seed": s, "test_matches": int(cl.max() + 1),
                         "map_batchspecific_minus_round": round(e1, 6), "ci_lo": round(l1, 6), "ci_hi": round(h1, 6),
                         "map_pooled_minus_round": round(e2, 6), "ci_lo_pooled": round(l2, 6),
                         "ci_hi_pooled": round(h2, 6)})
    out = pl.DataFrame(rows)
    print(out)
    summ = out.group_by("test_batch").agg(
        pl.col("map_batchspecific_minus_round").mean().round(6).alias("mean_batchspecific_map_increment"),
        pl.col("map_pooled_minus_round").mean().round(6).alias("mean_pooled_map_increment"),
        pl.col("test_matches").mean().round(1).alias("mean_test_matches"))
    print("\nSeed-averaged increments (negative = map helps):")
    print(summ)


if __name__ == "__main__":
    main()
