"""M4 models: does map identity or planted-site identity improve the bake-off model?

Same states, features, monotone constraints and XGBoost settings as scripts/model/bakeoff.py
(n_estimators 3000, eta 0.05, depth 4, min_child_weight 20, subsample 0.8, hist, seed 0,
early stopping 100 on the validation split). Two protocols:
  P1 primary split (match-level random, as the bake-off), early stopping on validation.
  P2 chronological split (as the bake-off), early stopping on validation.
  P3 five-fold match-grouped CV on all 1,521 kept matches, fixed 262 trees (ASSUMPTION: the
     baseline best iteration, reused for every variant so no variant gets extra tuning).
Site clusters (per map, k=2 on planter x,y,z) are refit inside each training set only.
Bootstrap: match-level, 1,000 draws, paired against V0 (baseline).
"""
import importlib.util
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

BK = r"G:\pureskill\csds-cs2\scripts\model\bakeoff.py"
SCR = (r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
       r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M4")
OUT = r"G:\pureskill\csds-cs2\scripts\swarm2\M4_models_results.json"
spec = importlib.util.spec_from_file_location("bakeoff_mod", BK)
bk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bk)
FEATS = list(bk.FEATS)
MONO_BASE = list(bk.MONO)
SEED = 0
MIN_MATCHES_MAJOR = 100
B_BOOT = 1000


def build_frames():
    S = pl.read_parquet(os.path.join(SCR, "states_2026.parquet")).with_columns(
        pl.col("round").cast(pl.Int64))
    R = pl.read_parquet(os.path.join(SCR, "rounds_2026.parquet")).with_columns(
        pl.col("round").cast(pl.Int64))
    counts = R.group_by("map_name").agg(pl.col("match_key").n_unique().alias("m"))
    major = sorted(counts.filter(pl.col("m") >= MIN_MATCHES_MAJOR)["map_name"].to_list())
    print("major maps (>= %d kept matches): %s" % (MIN_MATCHES_MAJOR, major))
    return S, R, major


def cluster_rounds(R, train_matches, major):
    """Fit k=2 per map on planted rounds of training matches; return {(match, round): cluster}
    for every planted round (train or not). Cluster 0 = larger training cluster."""
    assign = {}
    centres = {}
    planted = R.filter(pl.col("plant_x").is_not_null() & pl.col("plant_y").is_not_null()
                       & pl.col("plant_z").is_not_null())
    for mp in sorted(planted["map_name"].unique().to_list()):
        sub = planted.filter(pl.col("map_name") == mp)
        tr = sub.filter(pl.col("match_key").is_in(list(train_matches)))
        if tr.height < 20:
            continue
        Xtr = tr.select(["plant_x", "plant_y", "plant_z"]).to_numpy().astype(np.float64)
        km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(Xtr)
        sizes = np.bincount(km.labels_, minlength=2)
        big = int(np.argmax(sizes))
        order = [big, 1 - big]
        centres[mp] = km.cluster_centers_[order]
        Xall = sub.select(["plant_x", "plant_y", "plant_z"]).to_numpy().astype(np.float64)
        lab_raw = km.predict(Xall)
        lab = np.where(lab_raw == big, 0, 1)
        for (m, r), c in zip(sub.select(["match_key", "round"]).iter_rows(), lab):
            assign[(m, r)] = int(c)
    return assign, centres


def site_columns(S, R, assign, major):
    """Attach site features to the state table. Pre-plant states get -1 / 'pre'."""
    ca = pl.DataFrame({"match_key": pl.Series([k[0] for k in assign], dtype=pl.Utf8),
                       "round": pl.Series([k[1] for k in assign], dtype=pl.Int64),
                       "cluster": pl.Series(list(assign.values()), dtype=pl.Int64)})
    X = S.join(ca, on=["match_key", "round"], how="left")
    planted = pl.col("bomb_planted") == 1
    cl = pl.when(planted & pl.col("cluster").is_not_null()).then(pl.col("cluster")).otherwise(-1)
    X = X.with_columns(
        cl.alias("site01"),
        pl.when(planted).then(pl.col("plant_site_code")).otherwise(-1).alias("raw_site_code"),
        pl.when(planted).then(pl.col("plant_x")).otherwise(-99999.0).alias("plant_x_f"),
        pl.when(planted).then(pl.col("plant_y")).otherwise(-99999.0).alias("plant_y_f"),
        pl.when(planted).then(pl.col("plant_z")).otherwise(-99999.0).alias("plant_z_f"),
        pl.when(pl.col("map_name").is_in(major)).then(pl.col("map_name")).otherwise(pl.lit("other"))
        .alias("map_grp"),
    )
    X = X.with_columns(
        pl.when(planted & pl.col("cluster").is_not_null() & pl.col("map_name").is_in(major))
        .then(pl.col("map_name") + pl.lit("|") + pl.col("cluster").cast(pl.Utf8))
        .when(planted).then(pl.lit("other_post"))
        .otherwise(pl.lit("pre")).alias("msite"),
    )
    return X


def one_hot(values, levels):
    return np.column_stack([(values == lv).astype(np.float64) for lv in levels])


def make_matrix(X, variant, major):
    cols = [X[f].to_numpy().astype(np.float64) for f in FEATS]
    mono = list(MONO_BASE)
    names = list(FEATS)
    if variant in ("V1", "V4", "V7"):
        levels = major + ["other"]
        oh = one_hot(X["map_grp"].to_numpy(), levels)
        cols += [oh[:, i] for i in range(oh.shape[1])]
        names += [f"map={lv}" for lv in levels]
        mono += [0] * oh.shape[1]
    if variant == "V2":
        cols.append(X["site01"].to_numpy().astype(np.float64))
        names.append("site01")
        mono += [0]
    if variant in ("V3", "V4"):
        levels = [f"{m}|{c}" for m in major for c in (0, 1)] + ["other_post", "pre"]
        oh = one_hot(X["msite"].to_numpy(), levels)
        cols += [oh[:, i] for i in range(oh.shape[1])]
        names += [f"msite={lv}" for lv in levels]
        mono += [0] * oh.shape[1]
    if variant in ("V5", "V7"):
        for f in ("plant_x_f", "plant_y_f", "plant_z_f"):
            cols.append(X[f].to_numpy().astype(np.float64))
            names.append(f)
            mono += [0]
    if variant == "V6":
        cols.append(X["raw_site_code"].to_numpy().astype(np.float64))
        names.append("raw_site_code")
        mono += [0]
    if variant == "V8":
        cols.append(X["batch_1008"].to_numpy().astype(np.float64))
        names.append("batch_1008")
        mono += [0]
    return np.column_stack(cols), names, mono


VARIANTS = {
    "V0": "baseline: 10 bake-off features",
    "V1": "V0 + map one-hot (7 major maps + other)",
    "V2": "V0 + site01 (planted site cluster 0/1, no map; -1 pre-plant)",
    "V3": "V0 + map x site one-hot (14 map-site levels + other_post + pre)",
    "V4": "V0 + map one-hot + map x site one-hot",
    "V5": "V0 + planter x,y,z (post-plant; -99999 pre-plant)",
    "V6": "V0 + raw site_code numeric (post-plant; -1 pre-plant) [misuse check]",
    "V7": "V0 + map one-hot + planter x,y,z",
    "V8": "V0 + batch flag (match_date 2026-10-08) [batch-confound diagnostic]",
}


def fit_eval(Xtr, ytr, Xva, yva, Xte, mono, fixed_trees=None):
    params = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                  subsample=0.8, tree_method="hist", device="cpu", monotone_constraints=tuple(mono),
                  eval_metric="logloss", random_state=SEED, n_jobs=16)
    if fixed_trees is None:
        m = xgb.XGBClassifier(early_stopping_rounds=100, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bi = int(m.best_iteration)
        p = m.predict_proba(Xte, iteration_range=(0, bi + 1))[:, 1]
        return p, bi
    m = xgb.XGBClassifier(**{**params, "n_estimators": fixed_trees})
    m.fit(Xtr, ytr, verbose=False)
    return m.predict_proba(Xte)[:, 1], fixed_trees


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "logloss": float(log_loss(y, p)), "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p))}


def loss_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(match_ids, loss_a, loss_b, rng, B=B_BOOT):
    """Paired match-level bootstrap of mean loss difference (b minus a); negative = b better."""
    uniq, inv = np.unique(match_ids, return_inverse=True)
    M = len(uniq)
    sa = np.bincount(inv, weights=loss_a, minlength=M)
    sb = np.bincount(inv, weights=loss_b, minlength=M)
    n = np.bincount(inv, minlength=M).astype(np.float64)
    idx = rng.integers(0, M, size=(B, M))
    da = sa[idx].sum(1) / n[idx].sum(1)
    db = sb[idx].sum(1) / n[idx].sum(1)
    d = db - da
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d < 0))


def protocol_split(S, R, major, split_col, variants):
    train_m = set(R.filter(pl.col(split_col) == "train")["match_key"].to_list())
    val_m = set(R.filter(pl.col(split_col) == "val")["match_key"].to_list())
    test_m = set(R.filter(pl.col(split_col) == "test")["match_key"].to_list())
    assign, centres = cluster_rounds(R, train_m, major)
    X = site_columns(S, R, assign, major)
    y_all = X["label_ct_win"].to_numpy().astype(np.float64)
    mk = X["match_key"].to_numpy()
    tr = np.isin(mk, list(train_m))
    va = np.isin(mk, list(val_m))
    te = np.isin(mk, list(test_m))
    out = {}
    base_loss = None
    rng = np.random.default_rng(SEED)
    for v in variants:
        t0 = time.time()
        Xm, names, mono = make_matrix(X, v, major)
        p_te, bi = fit_eval(Xm[tr], y_all[tr], Xm[va], y_all[va], Xm[te], mono)
        y_te = y_all[te]
        met = metrics(y_te, p_te)
        met.update({"best_iteration": bi, "n_features": Xm.shape[1]})
        lv = loss_vec(y_te, p_te)
        if v == "V0":
            base_loss = (mk[te], lv)
            met["boot_vs_V0"] = None
        else:
            mean, lo, hi, pneg = boot_diff(base_loss[0], base_loss[1], lv, rng)
            met["boot_vs_V0"] = {"diff_mean": mean, "ci95": [lo, hi], "share_draws_better": pneg}
        met["seconds"] = round(time.time() - t0, 1)
        out[v] = met
        print(f"  {split_col} {v}: test logloss {met['logloss']:.5f} brier {met['brier']:.5f} "
              f"auc {met['auc']:.4f} best_iter {bi} feats {Xm.shape[1]} "
              f"diff_vs_V0 {met['boot_vs_V0']} ({met['seconds']}s)", flush=True)
    return out


def protocol_cv(S, R, major, variants, n_folds=5, trees=262):
    matches = sorted(R["match_key"].unique().to_list())
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(matches))
    fold_of = {matches[i]: int(j % n_folds) for j, i in enumerate(perm)}
    X0 = S.select(["match_key"]).to_series().to_numpy()
    fold_row = np.array([fold_of[m] for m in X0])
    out = {}
    oof = {v: np.zeros(S.height) for v in variants}
    for f in range(n_folds):
        tr_m = {m for m, g in fold_of.items() if g != f}
        te_m = {m for m, g in fold_of.items() if g == f}
        assign, _ = cluster_rounds(R, tr_m, major)
        X = site_columns(S, R, assign, major)
        y_all = X["label_ct_win"].to_numpy().astype(np.float64)
        mk = X["match_key"].to_numpy()
        tr = np.isin(mk, list(tr_m))
        te = np.isin(mk, list(te_m))
        for v in variants:
            Xm, names, mono = make_matrix(X, v, major)
            p_te, _ = fit_eval(Xm[tr], y_all[tr], None, None, Xm[te], mono, fixed_trees=trees)
            oof[v][te] = p_te
        print(f"  CV fold {f} done: train matches {len(tr_m)}, test matches {len(te_m)}", flush=True)
    y_all = S["label_ct_win"].to_numpy().astype(np.float64)
    mk = S["match_key"].to_numpy()
    base_mk, base_lv = mk, loss_vec(y_all, oof["V0"])
    rng = np.random.default_rng(SEED)
    for v in variants:
        met = metrics(y_all, oof[v])
        lv = loss_vec(y_all, oof[v])
        if v == "V0":
            met["boot_vs_V0"] = None
        else:
            mean, lo, hi, pneg = boot_diff(base_mk, base_lv, lv, rng)
            met["boot_vs_V0"] = {"diff_mean": mean, "ci95": [lo, hi], "share_draws_better": pneg}
        out[v] = met
        print(f"  CV {v}: OOF logloss {met['logloss']:.5f} brier {met['brier']:.5f} auc {met['auc']:.4f} "
              f"diff_vs_V0 {met['boot_vs_V0']}", flush=True)
    return out


def main():
    t0 = time.time()
    S, R, major = build_frames()
    print("states:", S.height, "rounds:", R.height, "matches:", R["match_key"].n_unique(), flush=True)
    variants = list(VARIANTS)
    results = {"variants": VARIANTS, "major_maps": major, "protocols": {}}
    print("== P1 primary split", flush=True)
    results["protocols"]["P1_primary"] = protocol_split(S, R, major, "split_primary", variants)
    print("== P2 chronological split", flush=True)
    results["protocols"]["P2_chrono"] = protocol_split(S, R, major, "split_chrono", variants)
    print("== P3 five-fold grouped CV, fixed 262 trees", flush=True)
    results["protocols"]["P3_cv5_fixed262"] = protocol_cv(S, R, major, variants)
    results["seconds_total"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", OUT, "total seconds", results["seconds_total"], flush=True)


if __name__ == "__main__":
    main()
