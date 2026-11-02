"""AUC, Brier, accuracy and calibration for the 57-input baseline and each temporal model, on the same de_mirage test seconds.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\model\\score_paired_metrics.py --temporal data\\temporal\\de_mirage_thgnn_lite_bestval_pred.npz data\\temporal\\de_mirage_gat_loadout_gru_pred.npz

Each temporal model is joined to the baseline by compare_paired_de_mirage.load_paired, so every row is the same second.
Point estimates: log loss, Brier score, AUC (rank-based), accuracy at 0.5, and expected calibration error (10 equal-width bins).
Differences (temporal minus baseline) for log loss, Brier and AUC come with a 1,000-resample match-cluster bootstrap (seed 0).
The log-loss bootstrap draws the same match samples as compare_paired_de_mirage.py, so its interval must agree with that script.
AUC is recomputed from the rank sums for each resample; it is checked against sklearn on the full sample.
Writes data/temporal/metrics_paired_de_mirage.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_paired_de_mirage as cpd  # noqa: E402  (load_paired, row_loss, OUT, MAP)

from sklearn.metrics import roc_auc_score  # noqa: E402

BINS = 10
N_BOOT = 1000


def ece(y, p, bins=BINS):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / y.size)


def groups(p):
    """Index of each row's unique score, ascending. Tied scores share a group."""
    uniq, grp = np.unique(p, return_inverse=True)
    return grp, uniq.size


def weighted_auc(y, grp, n_groups, w):
    """Mann-Whitney AUC with row weights; ties count one half. Groups must be sorted by ascending score."""
    pos = np.bincount(grp, weights=w * y, minlength=n_groups)
    neg = np.bincount(grp, weights=w * (1 - y), minlength=n_groups)
    neg_below = np.cumsum(neg) - neg
    return float(np.sum(pos * (neg_below + 0.5 * neg)) / (pos.sum() * neg.sum()))


def point_metrics(y, p):
    return {
        "log_loss": float(cpd.row_loss(y, p).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "auc": float(roc_auc_score(y, p)),
        "accuracy_at_0.5": float(((p >= 0.5) == (y == 1)).mean()),
        "ece_10_bins": ece(y, p),
        "mean_predicted": float(p.mean()),
        "observed_ct_win_rate": float(y.mean()),
    }


def score_model(path):
    paired, tag = cpd.load_paired(path)
    y = paired["y"].to_numpy()
    pb = paired["p_base"].to_numpy()
    pt = paired["p_temp"].to_numpy()
    match = paired["match_key"].to_numpy()
    uniq, inv = np.unique(match, return_inverse=True)
    n_match = uniq.size

    gb, nb = groups(pb)
    gt, nt = groups(pt)
    ones = np.ones_like(y)
    auc_b_fast = weighted_auc(y, gb, nb, ones)
    auc_b_sk = roc_auc_score(y, pb)
    assert abs(auc_b_fast - auc_b_sk) < 1e-9, (auc_b_fast, auc_b_sk)

    lb, lt = cpd.row_loss(y, pb), cpd.row_loss(y, pt)
    bb, bt = (pb - y) ** 2, (pt - y) ** 2
    cnt = np.bincount(inv, minlength=n_match).astype(float)
    s_lb = np.bincount(inv, weights=lb, minlength=n_match)
    s_lt = np.bincount(inv, weights=lt, minlength=n_match)
    s_bb = np.bincount(inv, weights=bb, minlength=n_match)
    s_bt = np.bincount(inv, weights=bt, minlength=n_match)

    rng = np.random.default_rng(0)
    d_ll, d_br, d_auc = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.integers(0, n_match, n_match)
        wm = np.bincount(pick, minlength=n_match).astype(float)
        total = (wm * cnt).sum()
        d_ll[b] = ((wm * s_lt).sum() - (wm * s_lb).sum()) / total
        d_br[b] = ((wm * s_bt).sum() - (wm * s_bb).sum()) / total
        w = wm[inv]
        d_auc[b] = weighted_auc(y, gt, nt, w) - weighted_auc(y, gb, nb, w)

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [float(lo), float(hi)]

    pb_metrics = point_metrics(y, pb)
    pt_metrics = point_metrics(y, pt)
    return {
        "temporal_model": tag,
        "paired_seconds": int(paired.height),
        "paired_matches": int(n_match),
        "baseline": pb_metrics,
        "temporal": pt_metrics,
        "difference_temporal_minus_baseline": {
            "log_loss": pt_metrics["log_loss"] - pb_metrics["log_loss"],
            "log_loss_ci95": ci(d_ll),
            "brier": pt_metrics["brier"] - pb_metrics["brier"],
            "brier_ci95": ci(d_br),
            "auc": pt_metrics["auc"] - pb_metrics["auc"],
            "auc_ci95": ci(d_auc),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", nargs="+", required=True, help="prediction npz files, one per temporal model")
    a = ap.parse_args()
    results = [score_model(p) for p in a.temporal]
    out = {
        "map": cpd.MAP,
        "baseline": "winprob_v6_drop_heavy (57 inputs)",
        "bootstrap_resamples": N_BOOT,
        "seed": 0,
        "reading": "differences are temporal minus baseline: positive log loss and Brier mean the temporal model is worse; "
                   "positive AUC means it ranks better",
        "models": results,
    }
    cpd.OUT.mkdir(parents=True, exist_ok=True)
    (cpd.OUT / "metrics_paired_de_mirage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for r in results:
        d = r["difference_temporal_minus_baseline"]
        print(f"{r['temporal_model']}: n={r['paired_seconds']} matches={r['paired_matches']}")
        print(f"  baseline  logloss {r['baseline']['log_loss']:.5f}  brier {r['baseline']['brier']:.5f}  "
              f"auc {r['baseline']['auc']:.4f}  acc {r['baseline']['accuracy_at_0.5']:.4f}  ece {r['baseline']['ece_10_bins']:.4f}")
        print(f"  temporal  logloss {r['temporal']['log_loss']:.5f}  brier {r['temporal']['brier']:.5f}  "
              f"auc {r['temporal']['auc']:.4f}  acc {r['temporal']['accuracy_at_0.5']:.4f}  ece {r['temporal']['ece_10_bins']:.4f}")
        print(f"  diff      logloss {d['log_loss']:+.5f} {['%+.5f' % x for x in d['log_loss_ci95']]}  "
              f"brier {d['brier']:+.5f} {['%+.5f' % x for x in d['brier_ci95']]}  "
              f"auc {d['auc']:+.4f} {['%+.4f' % x for x in d['auc_ci95']]}")


if __name__ == "__main__":
    main()
