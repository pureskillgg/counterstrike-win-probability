"""R4 evaluation demo. Read-only: reads data/state_sample, writes nothing.

Part A (real data, data/state_sample: 150 matches, 2,673 rounds, labelled train/test):
  constant base rate, logistic regression and HistGradientBoosting on 5 state features.
  Metrics on the test split: state-weighted log loss, round-weighted log loss, Brier, AUC,
  ECE with 10 equal-width bins, ECE with 15 equal-width bins (Guo et al. M=15), ECE with
  10 equal-mass bins. SE by i.i.d. state bootstrap, round-cluster bootstrap and match-cluster
  bootstrap, plus the design effect. Paired match-cluster bootstrap for LR minus HGB log loss.
  Equal-mass edges are fixed from the full test predictions (approximation inside bootstraps).

Part B (synthetic): what a random state-level split, a round-grouped split and a
  match-grouped split do to the test log loss when a roster-like match fingerprint and a
  round-constant fingerprint are available as features.

Part C (synthetic): finite-sample ECE for a perfectly calibrated predictor (equal-width vs
  equal-mass, 10/15/100 bins), and ECE, log loss and Brier for a constant base-rate predictor.

Run: pixi run python -I -X utf8 scripts/swarm/r4_eval_demo.py
"""
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

EPS = 1e-15
BASE = r"G:\pureskill\csds-cs2\data\state_sample"
FEATS = ["alive_diff", "score_diff", "seconds_since_live", "bomb_planted", "round_number"]
SCHEMES = [("ece_w10", "w", 10), ("ece_w15", "w", 15), ("ece_m10", "m", 10)]
N_BOOT = 2000
N_IID = 300


def codes(values):
    m = {}
    return np.array([m.setdefault(v, len(m)) for v in values], dtype=int)


def logloss_vec(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def edges_for(kind, B, p):
    if kind == "w":
        return np.linspace(0.0, 1.0, B + 1)
    inner = np.unique(np.quantile(p, np.linspace(0.0, 1.0, B + 1)[1:-1]))
    return np.unique(np.concatenate([[0.0], inner, [1.0]]))


def build_Z(y, p):
    """Per-state columns: log loss, squared error, count, then (y-p) placed in its bin per scheme."""
    N = len(y)
    blocks = [logloss_vec(y, p)[:, None], ((y - p) ** 2)[:, None], np.ones((N, 1))]
    sl, start = {}, 3
    for name, kind, B in SCHEMES:
        edges = edges_for(kind, B, p)
        nb = len(edges) - 1
        idx = np.digitize(p, edges[1:-1])
        Zb = np.zeros((N, nb))
        Zb[np.arange(N), idx] = y - p
        blocks.append(Zb)
        sl[name] = (start, start + nb)
        start += nb
    return np.hstack(blocks), sl


def stats(s, sl):
    n = s[2]
    out = {"logloss": s[0] / n, "brier": s[1] / n}
    for name, (a, b) in sl.items():
        out[name] = np.abs(s[a:b]).sum() / n  # ECE = sum_b |sum_{i in b}(y_i - p_i)| / N
    return out


def cluster_sums(cids, Z):
    M = np.zeros((cids.max() + 1, Z.shape[1]))
    np.add.at(M, cids, Z)
    return M


def cluster_boot(M, sl, rng, B=N_BOOT):
    C = M.shape[0]
    return [stats(M[rng.integers(0, C, C)].sum(0), sl) for _ in range(B)]


def iid_boot(Z, sl, rng, B=N_IID):
    N = Z.shape[0]
    return [stats(Z[rng.integers(0, N, N)].sum(0), sl) for _ in range(B)]


def sd(rows, key):
    return float(np.std([r[key] for r in rows], ddof=1))


def round_weighted_ll(Mr):
    return float(np.mean(Mr[:, 0] / Mr[:, 2]))


def load_state_sample():
    snap = pl.read_parquet(f"{BASE}\\positions_snapshots.parquet")

    def col(name):
        c = snap[name]
        if c.dtype == pl.Utf8:
            return c.is_not_null().to_numpy().astype(float)
        return c.cast(pl.Float64, strict=False).fill_null(0.0).to_numpy()

    X = np.column_stack([col(f) for f in FEATS])
    y = snap["ct_win"].cast(pl.Float64).to_numpy()
    split = snap["split"].to_numpy()
    mk = snap["match_key"].to_list()
    rnd = snap["round"].to_list()
    match_codes = codes(mk)
    round_codes = codes([f"{m}|{r}" for m, r in zip(mk, rnd)])
    return X, y, split, match_codes, round_codes


def part_a():
    print("=" * 70)
    print("PART A. Real data: data/state_sample, test split")
    X, y, split, mc, rc = load_state_sample()
    tr, te = split == "train", split == "test"
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    mc_te = codes(mc[te].tolist())
    rc_te = codes(rc[te].tolist())
    print(f"train states={tr.sum()} test states={te.sum()} "
          f"test matches={mc_te.max() + 1} test rounds={rc_te.max() + 1} "
          f"test ct_win rate={yte.mean():.4f}")

    preds = {
        "const_base_rate": np.full(te.sum(), ytr.mean()),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
                  .fit(Xtr, ytr).predict_proba(Xte)[:, 1],
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
               .fit(Xtr, ytr).predict_proba(Xte)[:, 1],
    }
    rng = np.random.default_rng(0)
    print(f"\n{'model':16s} {'LL':>7s} {'LL_rw':>7s} {'Brier':>7s} {'AUC':>6s} "
          f"{'ECEw10':>7s} {'ECEw15':>7s} {'ECEm10':>7s}")
    results = {}
    for name, p in preds.items():
        Z, sl = build_Z(yte, p)
        pt = stats(Z.sum(0), sl)
        Mr = cluster_sums(rc_te, Z)
        Mm = cluster_sums(mc_te, Z)
        ll_rw = round_weighted_ll(Mr)
        auc = roc_auc_score(yte, p)
        iid = iid_boot(Z, sl, rng)
        rb = cluster_boot(Mr, sl, rng)
        mb = cluster_boot(Mm, sl, rng)
        results[name] = (p, Z, sl, Mr, Mm)
        print(f"{name:16s} {pt['logloss']:7.4f} {ll_rw:7.4f} {pt['brier']:7.4f} {auc:6.4f} "
              f"{pt['ece_w10']:7.4f} {pt['ece_w15']:7.4f} {pt['ece_m10']:7.4f}")
        print(f"{'  SE iid':16s} " + " ".join(f"{k}={sd(iid, k):.4f}" for k in
              ["logloss", "brier", "ece_w10", "ece_m10"]))
        print(f"{'  SE round':16s} " + " ".join(f"{k}={sd(rb, k):.4f}" for k in
              ["logloss", "brier", "ece_w10", "ece_m10"]))
        print(f"{'  SE match':16s} " + " ".join(f"{k}={sd(mb, k):.4f}" for k in
              ["logloss", "brier", "ece_w10", "ece_m10"]))
        deff = (sd(mb, "logloss") / sd(iid, "logloss")) ** 2
        n_eff = Z.shape[0] / deff
        print(f"{'  design effect (match, logloss)':16s} {deff:.1f}  "
              f"effective states ~ {n_eff:.0f} of {Z.shape[0]}")
        lo, hi = np.percentile([r["logloss"] for r in mb], [2.5, 97.5])
        print(f"{'  match-cluster 95% CI LL':16s} [{lo:.4f}, {hi:.4f}]")

    # Paired comparison: LR minus HGB in state log loss, cluster bootstrap by match and by round.
    d = logloss_vec(yte, preds["logreg"]) - logloss_vec(yte, preds["hgb"])
    Zd = np.column_stack([d, np.ones(len(d))])
    Dm = cluster_sums(mc_te, Zd)
    Dr = cluster_sums(rc_te, Zd)
    diffs_m = np.array([Dm[rng.integers(0, Dm.shape[0], Dm.shape[0])].sum(0)
                        for _ in range(N_BOOT)])
    diffs_m = diffs_m[:, 0] / diffs_m[:, 1]
    diffs_r = np.array([Dr[rng.integers(0, Dr.shape[0], Dr.shape[0])].sum(0)
                        for _ in range(N_BOOT)])
    diffs_r = diffs_r[:, 0] / diffs_r[:, 1]
    mean_d = d.mean()
    print(f"\nPaired LL difference (logreg - hgb) = {mean_d:+.4f}")
    print(f"  match-cluster 95% CI [{np.percentile(diffs_m, 2.5):+.4f}, "
          f"{np.percentile(diffs_m, 97.5):+.4f}]  share of boot draws with diff<0: "
          f"{(diffs_m < 0).mean():.3f}")
    print(f"  round-cluster 95% CI [{np.percentile(diffs_r, 2.5):+.4f}, "
          f"{np.percentile(diffs_r, 97.5):+.4f}]  share of boot draws with diff<0: "
          f"{(diffs_r < 0).mean():.3f}")
    iid_d = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_IID)])
    print(f"  iid-state 95% CI [{np.percentile(iid_d, 2.5):+.4f}, {np.percentile(iid_d, 97.5):+.4f}]")


def synth_leak(seed, M=200, R=20, K=10):
    rng = np.random.default_rng(seed)
    s_m = rng.normal(0, 1, M)
    NR = M * R
    round_match = np.repeat(np.arange(M), R)
    eta = 0.6 * s_m[round_match] + rng.normal(0, 1, NR)
    y_r = (rng.random(NR) < 1 / (1 + np.exp(-eta))).astype(int)
    match_fp = np.column_stack([s_m + rng.normal(0, 0.5, M), rng.normal(0, 1, (M, 3))])
    round_fp = rng.normal(0, 1, (NR, 3))
    round_idx = np.repeat(np.arange(NR), K)
    match_idx = round_match[round_idx]
    y = y_r[round_idx]
    x = 0.4 * (2 * y - 1) + rng.normal(0, 1, NR * K)
    X_full = np.column_stack([x, match_fp[match_idx], round_fp[round_idx]])
    return X_full, y, round_idx, match_idx, NR, M


def part_b():
    print("=" * 70)
    print("PART B. Synthetic: split type vs test log loss (3 seeds, HGB on all features vs x-only LR)")
    rows = []
    for seed in range(3):
        X, y, ridx, midx, NR, M = synth_leak(seed)
        rng = np.random.default_rng(100 + seed)
        N = len(y)
        splits = {
            "state_random": rng.random(N) < 0.2,
            "round_grouped": rng.random(NR)[ridx] < 0.2,
            "match_grouped": rng.random(M)[midx] < 0.2,
        }
        for sname, te in splits.items():
            tr = ~te
            hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
            hgb.fit(X[tr], y[tr])
            p_full = hgb.predict_proba(X[te])[:, 1]
            ll_full = logloss_vec(y[te], p_full).mean()
            auc_full = roc_auc_score(y[te], p_full)
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            lr.fit(X[tr][:, :1], y[tr])
            p_x = lr.predict_proba(X[te][:, :1])[:, 1]
            ll_x = logloss_vec(y[te], p_x).mean()
            rows.append((seed, sname, ll_full, auc_full, ll_x))
    print(f"{'seed':>4s} {'split':15s} {'LL hgb(all)':>12s} {'AUC hgb':>8s} {'LL logreg(x)':>13s}")
    for r in rows:
        print(f"{r[0]:4d} {r[1]:15s} {r[2]:12.4f} {r[3]:8.4f} {r[4]:13.4f}")
    for sname in ["state_random", "round_grouped", "match_grouped"]:
        sub = [r for r in rows if r[1] == sname]
        print(f"mean {sname:14s} LL hgb(all)={np.mean([r[2] for r in sub]):.4f} "
              f"AUC={np.mean([r[3] for r in sub]):.4f} "
              f"LL logreg(x)={np.mean([r[4] for r in sub]):.4f}")


def part_c():
    print("=" * 70)
    print("PART C. Synthetic ECE on known-calibrated and constant predictors (300 reps)")
    rng = np.random.default_rng(7)
    reps = 300
    print("perfectly calibrated p ~ Beta(0.6, 0.6); y ~ Bernoulli(p)")
    print(f"{'n':>7s} {'scheme':>12s} {'mean ECE':>9s} {'sd':>7s}")
    for n in [200, 2000, 20000]:
        for name, kind, B in [("equal-width 10", "w", 10), ("equal-width 15", "w", 15),
                              ("equal-width 100", "w", 100), ("equal-mass 10", "m", 10),
                              ("equal-mass 100", "m", 100)]:
            vals = []
            for _ in range(reps):
                p = rng.beta(0.6, 0.6, n)
                y = (rng.random(n) < p).astype(float)
                e = edges_for(kind, B, p)
                idx = np.digitize(p, e[1:-1])
                S = np.bincount(idx, weights=y - p, minlength=len(e) - 1)
                vals.append(np.abs(S).sum() / n)
            print(f"{n:7d} {name:>15s} {np.mean(vals):9.4f} {np.std(vals):7.4f}")
    print("\nconstant predictor at base rate 0.5568, y ~ Bernoulli(0.5568)")
    print(f"{'n':>7s} {'ECE w10':>9s} {'log loss':>9s} {'Brier':>7s} {'AUC':>6s}")
    for n in [200, 2000, 32884]:
        ece_v, ll_v, br_v, auc_v = [], [], [], []
        for _ in range(reps):
            y = (rng.random(n) < 0.5568).astype(float)
            p = np.full(n, 0.5568)
            S = np.bincount(np.digitize(p, np.linspace(0, 1, 11)[1:-1]),
                            weights=y - p, minlength=10)
            ece_v.append(np.abs(S).sum() / n)
            ll_v.append(logloss_vec(y, p).mean())
            br_v.append(((y - p) ** 2).mean())
            auc_v.append(0.5 if y.min() == y.max() else roc_auc_score(y, p + 1e-9 * rng.random(n)))
        print(f"{n:7d} {np.mean(ece_v):9.4f} {np.mean(ll_v):9.4f} {np.mean(br_v):7.4f} "
              f"{np.mean(auc_v):6.3f}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
