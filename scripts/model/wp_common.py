"""Shared pieces for the combined (2024 + 2026) per-second win-probability experiments.

Works in the CUDA environment (polars + numpy + xgboost):
  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 <script>

  load_grid(store)            -> polars DataFrame (data/<store>/grid.parquet joined with match date)
  make_split(name, G, seed)   -> {"train": set, "val": set, "test": set} of match keys (match-level, never round-level)
  fit_xgb(Xtr, ytr, Xva, yva, mono, seed, device) -> (model, best_iteration)
  evaluate(...)               -> dict: log loss (all ticks, round-weighted), Brier, AUC, ECE, match-cluster 95% CI
Evaluation rules (advisor): split by match; log loss is the target metric; seconds inside a round are correlated so every
interval is a match-cluster bootstrap (1000 resamples, seed 0); a calibration table is returned with each evaluation.
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import random

import numpy as np
import polars as pl

ROOT = os.environ.get("CS2_ROOT", r"G:\pureskill\csds-cs2")  # the RunPod volume sets CS2_ROOT=/workspace/csds
EPS = 1e-15
N_BOOT = 1000
F6 = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"]
F8 = F6 + ["is_pistol_round", "is_overtime"]
F9 = F8 + ["round_number"]
F_SCORE = F9 + ["score_lead_diag"]  # the retired model's inputs (score lead as a feature): reference only
FEATURE_SETS = {"F6": F6, "F8": F8, "F9": F9, "F_SCORE": F_SCORE}
MONO_ALIVE = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1}


def load_grid(store="combined", cols=None):
    d = os.path.join(ROOT, "data", store)
    G = pl.read_parquet(os.path.join(d, "grid.parquet"), columns=cols)
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "match_date")
    return G.join(M, on="match_key", how="left")


def drop_roster_defect_rounds(G, store="combined"):
    """Leave out the rounds whose roster lacks a player (rounds.parquet missing_side is set).

    For such a round the side of the unlisted player is inferred, and in about 40% of them the inference disagrees with the event
    log (found by the train/serve parity check: 37 of 90 sampled rows in those rounds, 0 of 19,910 elsewhere). The living-player
    snapshot needs the right side of every player, so these rounds (0.4% of 2026) are excluded from every fit and evaluation that
    uses the snapshot features. A live service reads the sides from the game state and never has this defect."""
    R = pl.read_parquet(os.path.join(ROOT, "data", store, "rounds.parquet"), columns=["match_key", "round", "missing_side"])
    bad = R.filter(pl.col("missing_side").is_not_null()).select("match_key", "round")
    return G.join(bad, on=["match_key", "round"], how="anti")


def _holdout(keys, seed, test_frac=0.2, val_frac=0.1):
    ks = sorted(keys)
    rng = random.Random(seed)
    rng.shuffle(ks)
    n_test = round(test_frac * len(ks))
    test, pool = ks[:n_test], ks[n_test:]
    n_val = round(val_frac * len(pool))
    return set(pool[n_val:]), set(pool[:n_val]), set(test)


# ---- frozen, append-only assignment of matches to train / validation / test ----
# A split recomputed from "the matches in this table" changes for EVERY match when matches are added (the shuffle is over the whole sorted
# list), so a model trained after a data merge could have been validated or tested on matches an earlier model trained on, and results
# from different scripts (each building its own match table) were not on the same held-out set. The manifest fixes each match's set once:
# scripts/model/split_manifest.py --freeze writes it from the split the existing models used, --extend adds new matches by a hash rule and
# never changes an existing entry. make_split("random") uses the manifest when it exists.
HASH_TEST_FRAC, HASH_VAL_FRAC = 0.20, 0.08  # of all matches: test 20%, validation 8% (= 10% of the rest), train 72%
SPLIT_MANIFEST = os.environ.get("WP_SPLIT_MANIFEST") or os.path.join(ROOT, "data", "combined", "split_manifest.json")


def hash_assign(match_key, seed=0):
    """Deterministic set for a match nobody has assigned yet: depends on the key and the seed only, never on the other matches."""
    import hashlib
    u = int(hashlib.sha256(f"{seed}:{match_key}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return "test" if u < HASH_TEST_FRAC else "val" if u < HASH_TEST_FRAC + HASH_VAL_FRAC else "train"


def load_manifest(path=None):
    import json
    p = path or SPLIT_MANIFEST
    if str(p).lower() == "none" or not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        man = json.load(f)
    if not isinstance(man.get("assign"), dict) or not man["assign"]:
        raise ValueError(f"split manifest {p} has no assignments")
    return man


def frozen_split(keys, manifest, seed=0):
    """{"train","val","test"} for `keys`: the manifest's assignment, the hash rule for keys it does not list (reported once)."""
    assign = manifest["assign"]
    out = {"train": set(), "val": set(), "test": set()}
    unknown = 0
    for k in keys:
        s = assign.get(k)
        if s is None:
            unknown += 1
            s = hash_assign(k, manifest.get("seed", seed))
        out[s].add(k)
    if unknown:
        print(f"[split] {unknown} of {len(keys)} matches are not in the split manifest: assigned by the hash rule (run split_manifest.py --extend to record them)", flush=True)
    return out


def make_split(name, G, seed=0):
    """Match-level splits. random: the frozen manifest when data/combined/split_manifest.json exists (see above), else stratified by era
    (20% test, 10% of the rest validation per era; the rule the first models used, name "random_legacy" forces it).
    a_to_b ("2024_to_2026", "2026_to_2024"): train/val from era a, every match of era b is test.
    chrono: earliest 70% of matches by match_date train, next 10% validation, latest 20% test."""
    m = G.group_by("match_key").agg(pl.col("era").first(), pl.col("match_date").first())
    if name == "random":
        man = load_manifest()
        if man is not None:
            return frozen_split(m["match_key"].to_list(), man, seed)
        name = "random_legacy"
    if name == "random_legacy":
        tr, va, te = set(), set(), set()
        for era in sorted(m["era"].unique().to_list()):
            a, b, c = _holdout(m.filter(pl.col("era") == era)["match_key"].to_list(), seed)
            tr |= a
            va |= b
            te |= c
        return {"train": tr, "val": va, "test": te}
    if name in ("2024_to_2026", "2026_to_2024"):
        a, b = name.split("_to_")
        tr, va, _ = _holdout(m.filter(pl.col("era") == a)["match_key"].to_list(), seed, test_frac=0.0)
        return {"train": tr, "val": va, "test": set(m.filter(pl.col("era") == b)["match_key"].to_list())}
    if name == "chrono":
        # NOTE: match_date is the PROCESSING day for hand-uploaded matches (99.9% of platform "unknown", which since 2026-10-08 are mostly
        # FACEIT), so this split puts the latest-processed FACEIT matches in test and almost none in train (audit Y2: test 77% FACEIT-server,
        # train 2%). Read it as a source-shift test (train mostly Valve matchmaking, test mostly FACEIT), NOT as a time test. Use
        # "chrono_within_source" for time order inside each source.
        o = m.sort(["match_date", "match_key"])["match_key"].to_list()
        n = len(o)
        n_tr, n_va = round(0.7 * n), round(0.1 * n)
        return {"train": set(o[:n_tr]), "val": set(o[n_tr:n_tr + n_va]), "test": set(o[n_tr + n_va:])}
    if name == "chrono_within_source":
        if "source" not in G.columns:
            raise ValueError("chrono_within_source needs a `source` column (see attach_source)")
        ms = G.group_by("match_key").agg(pl.col("source").first(), pl.col("match_date").first())
        tr, va, te = set(), set(), set()
        for src in ms["source"].unique().to_list():
            o = ms.filter(pl.col("source") == src).sort(["match_date", "match_key"])["match_key"].to_list()
            n = len(o)
            n_tr, n_va = round(0.7 * n), round(0.1 * n)
            tr |= set(o[:n_tr])
            va |= set(o[n_tr:n_tr + n_va])
            te |= set(o[n_tr + n_va:])
        return {"train": tr, "val": va, "test": te}
    raise ValueError(name)


def attach_source(G, store="combined"):
    """Adds `source` (faceit_server | valve_server | other) from the header's server_name. The data owner's rule: a FACEIT match
    has server_name beginning "FACEIT.com"; platform alone is not enough (hand uploads are platform "unknown")."""
    import glob
    pages = sorted(glob.glob(os.path.join(ROOT, "tomes", store, "tome", "csds", "header", "dataframe_*")))
    H = pl.concat([pl.read_parquet(f, columns=["key", "server_name"]) for f in pages], how="vertical_relaxed").rename({"key": "match_key"})
    H = H.with_columns(pl.when(pl.col("server_name").str.starts_with("FACEIT")).then(pl.lit("faceit_server"))
                       .when(pl.col("server_name").str.starts_with("Valve")).then(pl.lit("valve_server")).otherwise(pl.lit("other")).alias("source"))
    return G.join(H.select("match_key", "source").unique("match_key"), on="match_key", how="left")


def arrays(G, keys, feats, stride=1):
    """Rows of the given matches; stride > 1 keeps every stride-th second (training convenience: adjacent seconds are nearly
    identical, so halving them barely changes a fit and halves its memory). Evaluation always uses stride 1."""
    sub = G.filter(pl.col("match_key").is_in(sorted(keys)))
    if stride > 1:
        sub = sub.filter(pl.col("second") % stride == 0)
    X = sub.select(feats).to_numpy().astype(np.float32)
    y = sub["label_ct_win"].to_numpy().astype(np.int64)
    return sub, X, y


def fit_xgb(Xtr, ytr, Xva, yva, mono=None, seed=0, device="cuda", depth=4, eta=0.05, mcw=20, **extra):
    import xgboost as xgb
    params = dict(n_estimators=3000, learning_rate=eta, max_depth=depth, min_child_weight=mcw, subsample=0.8,
                  tree_method="hist", device=device, early_stopping_rounds=100, eval_metric="logloss", random_state=seed)
    params.update(extra)
    if mono is not None:
        params["monotone_constraints"] = tuple(mono)
    last = None
    for attempt in range(3):
        try:
            m = xgb.XGBClassifier(**params)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return m, int(m.best_iteration)
        except Exception as exc:  # the GPU is shared with another job; retry transient CUDA errors
            last = exc
            import time
            time.sleep(20)
    raise last


def predict(m, best, X):
    return m.predict_proba(X, iteration_range=(0, best + 1))[:, 1]


# Opt-in (WP_EXTRA_MONO=1): physically unambiguous directions for features added later. More own hit points never lowers a
# side's chance; with the bomb not planted, more elapsed time never lowers CT's chance (the clock expiring is a CT win).
MONO_EXTRA = {"hp_ct": 1, "hp_t": -1, "st_hp_ct": 1, "st_hp_t": -1, "t_unplanted": 1}


def mono_for(feats):
    m = dict(MONO_ALIVE)
    if os.environ.get("WP_EXTRA_MONO", "0") == "1":
        m.update(MONO_EXTRA)
    return [m.get(f, 0) for f in feats]


def ll_vec(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y, p):
    y = np.asarray(y).astype(bool)
    order = np.argsort(p, kind="mergesort")
    ps = np.asarray(p)[order]
    _, start, cnt = np.unique(ps, return_index=True, return_counts=True)
    avg = start + (cnt + 1) / 2.0
    ranks = np.empty(len(ps))
    ranks[order] = np.repeat(avg, cnt)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ece10(y, p):
    b = np.minimum((np.asarray(p) * 10).astype(int), 9)
    e = 0.0
    for i in range(10):
        m = b == i
        if m.any():
            e += m.mean() * abs(float(np.mean(y[m])) - float(np.mean(np.asarray(p)[m])))
    return float(e)


def fingerprint(*parts):
    """Short hash of everything a saved result depends on (features, constraints, parameters, split keys, ...). A checkpoint is reused only
    when its fingerprint matches, so a rerun after a change in settings trains again instead of mixing old and new predictions."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def dump_json(obj, path):
    """Write JSON atomically (temp file in the same folder, then os.replace): an interrupted run never leaves a truncated results file."""
    import json
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def ece_eqmass(y, p, bins=20):
    """ECE with bins of EQUAL COUNT (sorted by prediction): no bin is empty or tiny, unlike equal-width bins at the extremes."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    e = 0.0
    for part in np.array_split(order, bins):
        if len(part):
            e += len(part) / len(p) * abs(float(y[part].mean()) - float(p[part].mean()))
    return float(e)


def smooth_ece(y, p, grid=1000):
    """Smooth ECE (Blasiok and Nakkiran, 2024): the integral over t of |sum_i K(t - p_i)(y_i - p_i)| / n with a Gaussian kernel K of width
    sigma, reflected at 0 and 1, sigma chosen as the fixed point smECE(sigma) = sigma (found by bisection). No binning choice."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    r = np.asarray(y, dtype=np.float64) - p
    n = len(p)
    edges = np.linspace(0.0, 1.0, grid + 1)
    h = np.histogram(p, bins=edges, weights=r)[0]
    centers = (edges[:-1] + edges[1:]) / 2.0

    def sme(sigma):
        half = min(int(np.ceil(4.0 * sigma * grid)) + 1, grid)
        k = np.exp(-0.5 * ((np.arange(-half, half + 1) / grid) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))  # Gaussian DENSITY in t
        padded = np.concatenate([h[:half][::-1], h, h[-half:][::-1]])  # reflect the mass at the borders
        num = np.convolve(padded, k, mode="same")[half:half + grid]  # num[t] = sum_i (y_i - p_i) K(t - p_i)
        return float(np.abs(num).sum() / grid / n)  # integral over t of |num| (cell width 1/grid), divided by n

    lo, hi = 1e-3, 0.5
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if sme(mid) > mid:
            lo = mid
        else:
            hi = mid
    return float(sme((lo + hi) / 2.0))


def _ratio(num, den, mid, idx):
    M = idx.shape[1]
    sn = np.bincount(mid, weights=num, minlength=M)
    sd = np.bincount(mid, weights=den, minlength=M)
    return sn[idx].sum(axis=1) / np.maximum(sd[idx].sum(axis=1), 1e-12), float(sn.sum() / sd.sum())


def _ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def evaluate(sub, y, p, ref=None, bins=10):
    """sub: polars frame of the test rows (match_key, w_round). ref: optional reference predictions for a paired difference."""
    mk = sub["match_key"].to_numpy()
    uniq, mid = np.unique(mk, return_inverse=True)
    M = len(uniq)
    idx = np.random.default_rng(0).integers(0, M, size=(N_BOOT, M))
    wr = sub["w_round"].to_numpy().astype(np.float64)
    one = np.ones(len(y))
    ll = ll_vec(p, y)
    reps, pt = _ratio(ll, one, mid, idx)
    repr_, ptr = _ratio(ll * wr, wr, mid, idx)
    out = {"n_rows": int(len(y)), "n_matches": int(M), "logloss": pt, "logloss_ci95": _ci(reps),
           "logloss_round_weighted": ptr, "logloss_round_weighted_ci95": _ci(repr_),
           "brier": float(np.mean((np.asarray(p) - y) ** 2)), "auc": auc(y, p), "ece10": ece10(y, p),
           "ece20_equal_count": ece_eqmass(y, p, 20), "smooth_ece": smooth_ece(y, p)}
    # match-cluster intervals for Brier and ECE10 too (a point estimate of ECE hides how noisy it is; the audit asked for it): per match and
    # bin the sums of y and p, then ECE of a resample = sum over bins |sum y - sum p| / n
    yy, pp = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    reps_b, _ = _ratio((pp - yy) ** 2, one, mid, idx)
    b10 = np.minimum((pp * 10).astype(int), 9)
    Sy = np.stack([np.bincount(mid, weights=yy * (b10 == i), minlength=M) for i in range(10)], axis=1)
    Sp = np.stack([np.bincount(mid, weights=pp * (b10 == i), minlength=M) for i in range(10)], axis=1)
    n_rep = np.bincount(mid, minlength=M)[idx].sum(axis=1).astype(np.float64)
    reps_e = np.abs(Sy[idx].sum(axis=1) - Sp[idx].sum(axis=1)).sum(axis=1) / n_rep
    out["brier_ci95"] = _ci(reps_b)
    out["ece10_ci95"] = _ci(reps_e)
    if ref is not None:
        d = ll - ll_vec(ref, y)
        rd, dpt = _ratio(d, one, mid, idx)
        out["delta_vs_ref"] = dpt
        out["delta_vs_ref_ci95"] = _ci(rd)
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    cal = []
    for i in range(bins):
        m = b == i
        n = int(m.sum())
        if n:
            rate_reps, rate = _ratio(y * m, m.astype(np.float64), mid, idx)
            cal.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "mean_pred": float(np.mean(p[m])),
                        "observed": float(rate), "obs_ci95": _ci(rate_reps)})
    out["calibration"] = cal
    return out


class RidgeLogit:
    """Standardised L2 logistic regression (C = 1, intercept unpenalised), Newton steps, numpy only."""

    def fit(self, X, y, C=1.0, iters=30):
        X = np.asarray(X, dtype=np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        lam = np.full(Z.shape[1], 1.0 / C)
        lam[0] = 0.0
        w = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w)))
            g = Z.T @ (p - y) + lam * w
            H = (Z * (p * (1.0 - p))[:, None]).T @ Z + np.diag(lam)
            step = np.linalg.solve(H, g)
            w = w - step
            if np.abs(step).max() < 1e-9:
                break
        self.w = w
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1.0 / (1.0 + np.exp(-(Z @ self.w)))
