"""Explainability for the 83-input XGBoost candidate (validation-chosen, sprint cell full_d4_e0.03_m20_s0).

Model: 59 base inputs + GEO (16) + VIEW (4) + BLIND (4); depth 4, eta 0.03, min_child_weight 20, subsample 0.8, max_bin 256,
early stopping 100 on the early-stop rows, 3,000-round cap, seed 0. Row split and settings follow
experiments/sprint/sprint_worker.py (fit_cell_prepared). The sprint kept predictions but not the booster, so stage `train`
refits it here on CPU (this Windows xgboost build has no CUDA) and checks the validation log loss and predictions against the
sprint's saved predictions. (The pixi default environment used here is CPU xgboost; the GPU sprint result is not refit here.)

Stages (run in order):
  train    refit, save the booster, compare with the sprint predictions
  explain  TreeSHAP (xgboost pred_contribs) on a random validation sample; permutation importance on all validation rows
  report   tables (CSV) and figures (PNG) under docs/explainability/

Explainability (TreeSHAP) and importance (permutation, black box) are reported separately on purpose.
Large files (matrix, booster, sampled arrays) stay in WORK, outside git.

Run with the pixi environment that has xgboost:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\xgb83_explain.py --stage train
"""
import argparse
import json
import os
import time

import numpy as np
import polars as pl
import xgboost as xgb

MDIR = os.environ.get("CS2_EXPLAIN_MATRIX", "C:/Users/13144/sprint/matrix_trainctr")  # matrix from sprint_prepare.py (not in git)
WORK = os.environ.get("CS2_EXPLAIN_WORK", "C:/Users/13144/sprint/explain83")  # booster, sampled arrays (not in git)
SPRINT_PRED = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/preds/full_d4_e0.03_m20_s0.parquet"
SPRINT_RESULT = "G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/results/full_d4_e0.03_m20_s0.json"
DOCS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "explainability"))

PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.03, "max_depth": 4,
          "min_child_weight": 20, "subsample": 0.8, "tree_method": "hist", "seed": 0}
MONO = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1}  # same as wp_common.MONO_ALIVE
MAX_BIN = 256
SHAP_N = 100_000  # validation rows for TreeSHAP
REPEATS_GROUP = 2  # permutation repeats per bundle
REPEATS_FEATURE = 1  # permutation repeats per single feature
RNG_SEED = 20261105
EPS = 1e-15

# ---- bundles (the 83 inputs, each in exactly one group) -----------------------------------------------------------
ROUND_STATE = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"]
MAPS = ["map_is_ancient", "map_is_anubis", "map_is_cache", "map_is_dust2", "map_is_inferno", "map_is_mirage", "map_is_nuke"]


def side_groups(s):
    return {
        f"LOADOUT_{s.upper()}_HEALTH_ARMOR": [f"st_hp_{s}", f"st_armor_n_{s}", f"st_armor_sum_{s}", f"st_helmet_n_{s}",
                                              f"st_kit_n_{s}", f"st_defusing_{s}"],
        f"LOADOUT_{s.upper()}_WEAPONS": [f"st_{w}_{s}" for w in
                                         ["ak", "m4", "awp", "rifle", "sniper_other", "smg", "heavy", "no_primary"]],
        f"LOADOUT_{s.upper()}_GRENADES": [f"st_{g}_{s}" for g in ["flash", "smoke", "he", "fire", "decoy"]],
        f"LOADOUT_{s.upper()}_EQUIP": [f"st_equip_{s}"],
    }


GROUPS = {"ROUND_STATE": ROUND_STATE, "MAP": MAPS}
GROUPS.update(side_groups("ct"))
GROUPS.update(side_groups("t"))
GROUPS["LOADOUT_DIFF"] = ["st_hp_diff", "st_rifle_diff", "st_equip_diff", "st_armor_n_diff", "st_awp_diff", "st_smg_diff"]
GROUPS["GEO_SITES_SPREAD"] = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_nearest_site_mean", "t_nearest_site_mean",
                              "c_spread", "t_spread", "ct_t_centroid_dist"]
GROUPS["GEO_CONTACT"] = ["min_enemy_dist", "c_in_contact", "t_in_contact"]
GROUPS["BOMB_GEO"] = ["c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist", "defuse_slack"]
GROUPS["VIEW"] = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
GROUPS["BLIND"] = ["c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]
POSITION_GROUPS = ["GEO_SITES_SPREAD", "GEO_CONTACT", "BOMB_GEO", "VIEW", "BLIND"]
LOADOUT_GROUPS = [g for g in GROUPS if g.startswith("LOADOUT_")]

# Flags from the team's review notes on feature quality. They are labels for reading the results only; they do not change
# the model. The checklist asked for st_armor_sum_ct/t and the sees-enemy/exposed inputs to be removed; the 83-input model
# keeps them. This is disclosed, and the 77-input variant (without them) is the fallback.
NOTES = {
    "st_armor_sum_ct": "in the 83 model; checklist asked to drop it (disclosed; 77 variant drops it)",
    "st_armor_sum_t": "in the 83 model; checklist asked to drop it (disclosed; 77 variant drops it)",
    "c_sees_enemy": "in the 83 model; checklist asked to remove (enemy-spotted output unreliable; disclosed)",
    "t_sees_enemy": "in the 83 model; checklist asked to remove (enemy-spotted output unreliable; disclosed)",
    "c_exposed": "in the 83 model; checklist asked to remove with sees-enemy (disclosed)",
    "t_exposed": "in the 83 model; checklist asked to remove with sees-enemy (disclosed)",
    "c_blind_n": "built from S3 player_blind events",
    "t_blind_n": "built from S3 player_blind events",
    "c_blind_sum": "built from S3 player_blind events",
    "t_blind_sum": "built from S3 player_blind events",
    "defuse_slack": "in the 83 model; crude (defuse race time to reach and defuse)",
    "st_heavy_ct": "in the 83 model; dropped only in the v6 variant",
    "st_heavy_t": "in the 83 model; dropped only in the v6 variant",
}


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def load_matrix():
    meta = json.load(open(os.path.join(MDIR, "meta.json")))
    feats = meta["feats_all"]
    assert len(feats) == 83 and len(set(feats)) == 83 and feats[:59] == meta["base"]
    covered = [f for cols in GROUPS.values() for f in cols]
    assert sorted(covered) == sorted(feats), "bundles do not cover the 83 inputs exactly once"
    X = np.load(os.path.join(MDIR, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(MDIR, "y.npy"), mmap_mode="r")
    assert X.shape == (meta["rows"], 83) and y.shape == (meta["rows"],)
    return meta, feats, X, y


def stage_train():
    os.makedirs(WORK, exist_ok=True)
    meta, feats, X, y = load_matrix()
    a1, a2 = meta["n_fit"], meta["n_fit"] + meta["n_es"]
    mono = tuple(MONO.get(f, 0) for f in feats)
    t0 = time.time()
    dtrain = xgb.QuantileDMatrix(np.asarray(X[:a1]), np.asarray(y[:a1]), max_bin=MAX_BIN)
    dval = xgb.QuantileDMatrix(np.asarray(X[a1:a2]), np.asarray(y[a1:a2]), ref=dtrain, max_bin=MAX_BIN)
    print(f"quantised {a1:,} fit and {a2 - a1:,} early-stop rows in {time.time() - t0:.0f}s", flush=True)

    params = dict(PARAMS, device="cpu", monotone_constraints=mono, nthread=os.cpu_count())
    t0 = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=3000, evals=[(dval, "validation_0")],
                    early_stopping_rounds=100, verbose_eval=100)
    fit_s = time.time() - t0
    best = int(bst.best_iteration)
    bst.save_model(os.path.join(WORK, "xgb83_s0.json"))
    print(f"best iteration {best}, fit {fit_s:.0f}s", flush=True)

    a2_rows = slice(a2, None)
    Xv = np.asarray(X[a2_rows], dtype=np.float32)
    yv = np.asarray(y[a2_rows], dtype=np.float64)
    p = bst.predict(xgb.DMatrix(Xv), iteration_range=(0, best + 1))
    ll = logloss(p, yv)

    vk = pl.read_parquet(os.path.join(MDIR, "val_keys.parquet"))
    sp = pl.read_parquet(SPRINT_PRED)
    assert sp.height == vk.height == len(yv), "row counts differ"
    for c in ["match_key", "round", "second"]:
        assert (sp[c].to_numpy() == vk[c].to_numpy()).all(), f"sprint prediction rows differ in {c}"
    d = np.abs(sp["p"].to_numpy() - p)
    sprint_ll = json.load(open(SPRINT_RESULT))["val_logloss"]
    summary = {"best_iteration": best, "sprint_best_iteration": json.load(open(SPRINT_RESULT))["best_iteration"],
               "fit_seconds": round(fit_s, 1), "val_logloss_refit": ll, "val_logloss_sprint": sprint_ll,
               "pred_max_abs_diff_vs_sprint": float(d.max()), "pred_mean_abs_diff_vs_sprint": float(d.mean()),
               "n_validation": int(len(yv)), "features": feats}
    json.dump(summary, open(os.path.join(WORK, "train_meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "features"}, indent=1), flush=True)


def stage_explain():
    meta, feats, X, y = load_matrix()
    a2 = meta["n_fit"] + meta["n_es"]
    tm = json.load(open(os.path.join(WORK, "train_meta.json")))
    best = tm["best_iteration"]
    iters = (0, best + 1)
    bst = xgb.Booster()
    bst.load_model(os.path.join(WORK, "xgb83_s0.json"))

    Xv = np.asarray(X[a2:], dtype=np.float32)
    yv = np.asarray(y[a2:], dtype=np.float64)
    n = Xv.shape[0]
    rng = np.random.default_rng(RNG_SEED)

    # 1) TreeSHAP on a random validation sample (explainability)
    s_idx = np.sort(rng.choice(n, size=SHAP_N, replace=False))
    dS = xgb.DMatrix(Xv[s_idx])
    t0 = time.time()
    contrib = bst.predict(dS, pred_contribs=True, iteration_range=iters)
    z = bst.predict(dS, output_margin=True, iteration_range=iters)
    print(f"TreeSHAP on {SHAP_N:,} rows in {time.time() - t0:.0f}s", flush=True)
    add_err = float(np.abs(contrib.sum(axis=1) - z).max())
    assert add_err < 1e-3, f"SHAP additivity error {add_err}"
    np.savez(os.path.join(WORK, "shap_sample.npz"), phi=contrib[:, :83].astype(np.float32), bias=contrib[:, 83],
             z=z.astype(np.float32), idx=s_idx)
    np.save(os.path.join(WORK, "xs.npy"), Xv[s_idx])
    print(f"additivity max error {add_err:.2e}", flush=True)

    # 2) Permutation importance on all validation rows (black-box importance, model-agnostic)
    dV = xgb.DMatrix(Xv)
    base_p = bst.predict(dV, iteration_range=iters)
    base_ll = logloss(base_p, yv)
    print(f"validation log loss (refit): {base_ll:.6f}", flush=True)
    jobs = [("bundle", g, [feats.index(f) for f in cols], REPEATS_GROUP) for g, cols in GROUPS.items()]
    jobs += [("feature", f, [feats.index(f)], REPEATS_FEATURE) for f in feats]
    rows = []
    t0 = time.time()
    for kind, name, cols, reps in jobs:
        deltas = []
        for _ in range(reps):
            perm = rng.permutation(n)
            Xp = Xv.copy()
            Xp[:, cols] = Xv[perm][:, cols]
            p = bst.predict(xgb.DMatrix(Xp), iteration_range=iters)
            deltas.append(logloss(p, yv) - base_ll)
            del Xp
        rows.append({"kind": kind, "name": name, "perm_delta_logloss_mean": float(np.mean(deltas)),
                     "perm_delta_logloss_min": float(np.min(deltas)), "perm_delta_logloss_max": float(np.max(deltas)),
                     "repeats": reps})
    pl.DataFrame(rows).write_csv(os.path.join(WORK, "perm_importance.csv"))
    print(f"permutation importance: {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    json.dump({"base_logloss_all_validation": base_ll, "n_validation": int(n), "best_iteration": best,
               "shap_rows": SHAP_N, "repeats_group": REPEATS_GROUP, "repeats_feature": REPEATS_FEATURE},
              open(os.path.join(WORK, "explain_meta.json"), "w"), indent=1)


def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tables = os.path.join(DOCS, "tables")
    figs = os.path.join(DOCS, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figs, exist_ok=True)
    meta, feats, X, y = load_matrix()
    S = np.load(os.path.join(WORK, "shap_sample.npz"))
    phi = S["phi"].astype(np.float64)
    z = S["z"].astype(np.float64)
    Xs = np.load(os.path.join(WORK, "xs.npy")).astype(np.float64)
    p = sig(z)
    perm = pl.read_csv(os.path.join(WORK, "perm_importance.csv"))
    perm_b = {r["name"]: r["perm_delta_logloss_mean"] for r in perm.filter(pl.col("kind") == "bundle").iter_rows(named=True)}
    perm_f = {r["name"]: r["perm_delta_logloss_mean"] for r in perm.filter(pl.col("kind") == "feature").iter_rows(named=True)}
    grp_of = {f: g for g, cols in GROUPS.items() for f in cols}

    # per-input: leave-one-input-out effect in probability points, row by row: 100*(sigmoid(z) - sigmoid(z - phi_f))
    pp_f = 100.0 * (p[:, None] - sig(z[:, None] - phi))
    feat_rows = []
    for j, f in enumerate(feats):
        feat_rows.append({"feature": f, "bundle": grp_of[f], "note": NOTES.get(f, ""),
                          "mean_abs_shap_logit": float(np.abs(phi[:, j]).mean()),
                          "mean_abs_effect_pp": float(np.abs(pp_f[:, j]).mean()),
                          "mean_signed_effect_pp": float(pp_f[:, j].mean()),
                          "nan_rate_sample": float(np.isnan(Xs[:, j]).mean()),
                          "perm_delta_logloss": perm_f.get(f)})
    fi = pl.DataFrame(feat_rows).with_columns(
        pl.col("mean_abs_effect_pp").rank("ordinal", descending=True).cast(pl.Int64).alias("rank_effect_pp"),
        pl.col("perm_delta_logloss").rank("ordinal", descending=True).cast(pl.Int64).alias("rank_perm")
    ).sort("mean_abs_effect_pp", descending=True)
    fi.write_csv(os.path.join(tables, "feature_importance_and_effects.csv"))

    # bundles: the sum of member SHAP values is exact (SHAP is additive), so the bundle effect is the same as removing the bundle
    grp_rows = []
    for g, cols in GROUPS.items():
        idx = [feats.index(f) for f in cols]
        phi_g = phi[:, idx].sum(axis=1)
        pp_g = 100.0 * (p - sig(z - phi_g))
        grp_rows.append({"bundle": g, "n_inputs": len(cols), "mean_abs_bundle_shap_logit": float(np.abs(phi_g).mean()),
                         "mean_abs_effect_pp": float(np.abs(pp_g).mean()), "mean_signed_effect_pp": float(pp_g.mean()),
                         "perm_delta_logloss": perm_b.get(g), "inputs": ", ".join(cols)})
    for name, members in [("ALL_POSITION (GEO+VIEW+BLIND)", POSITION_GROUPS), ("ALL_LOADOUT", LOADOUT_GROUPS)]:
        cols = [f for m in members for f in GROUPS[m]]
        idx = [feats.index(f) for f in cols]
        phi_g = phi[:, idx].sum(axis=1)
        pp_g = 100.0 * (p - sig(z - phi_g))
        grp_rows.append({"bundle": name, "n_inputs": len(cols), "mean_abs_bundle_shap_logit": float(np.abs(phi_g).mean()),
                         "mean_abs_effect_pp": float(np.abs(pp_g).mean()), "mean_signed_effect_pp": float(pp_g.mean()),
                         "perm_delta_logloss": None, "inputs": ", ".join(cols)})
    gi = pl.DataFrame(grp_rows).sort("mean_abs_effect_pp", descending=True)
    gi.write_csv(os.path.join(tables, "bundle_importance_and_effects.csv"))

    # dependence: binned mean effect (pp) against the input value, for the top inputs
    dep_rows = []
    top = fi["feature"].head(10).to_list()
    for f in top:
        j = feats.index(f)
        x = Xs[:, j]
        ok = ~np.isnan(x)
        uniq = np.unique(x[ok])
        if len(uniq) <= 12:
            bins = [(u, u) for u in uniq]
            assign = [(x == u) for u, _ in bins]
        else:
            edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 21)))
            bins = list(zip(edges[:-1], edges[1:]))
            assign = []
            for k, (lo, hi) in enumerate(bins):
                last = k == len(bins) - 1
                assign.append(ok & (x >= lo) & (x <= hi if last else x < hi))
        for (lo, hi), m in zip(bins, assign):
            if m.sum() == 0:
                continue
            dep_rows.append({"feature": f, "x_lo": float(lo), "x_hi": float(hi), "n": int(m.sum()),
                             "mean_value": float(x[m].mean()), "mean_signed_effect_pp": float(pp_f[m, j].mean()),
                             "mean_abs_effect_pp": float(np.abs(pp_f[m, j]).mean())})
        if (~ok).any():
            m = ~ok
            dep_rows.append({"feature": f, "x_lo": None, "x_hi": None, "n": int(m.sum()), "mean_value": None,
                             "mean_signed_effect_pp": float(pp_f[m, j].mean()),
                             "mean_abs_effect_pp": float(np.abs(pp_f[m, j]).mean())})
    pl.DataFrame(dep_rows).write_csv(os.path.join(tables, "dependence_top10.csv"))

    # ---- figures ----
    fam_color = {"ROUND_STATE": "#4C72B0", "MAP": "#8172B2"}
    for g in GROUPS:
        if g.startswith("LOADOUT_"):
            fam_color[g] = "#DD8452"
    for g in POSITION_GROUPS:
        fam_color[g] = "#55A868"
    bars = gi.filter(~pl.col("bundle").str.starts_with("ALL_"))
    names = bars["bundle"].to_list()[::-1]
    vals = bars["mean_abs_effect_pp"].to_list()[::-1]
    cols_ = [fam_color.get(n, "#999999") for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.2))
    ax.barh(names, vals, color=cols_)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("mean |effect| on P(CT wins), percentage points per second")
    ax.set_title("What each bundle adds to P(CT wins), TreeSHAP", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "bundle_effects.png"), dpi=130)
    plt.close(fig)

    top25 = fi.head(25)
    names = top25["feature"].to_list()[::-1]
    vals = top25["mean_abs_effect_pp"].to_list()[::-1]
    cols_ = [fam_color.get(grp_of[n], "#999999") for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(names) + 1.2))
    ax.barh(names, vals, color=cols_)
    flagged = {n for n in names if n in NOTES}
    for i, n in enumerate(names):
        mark = "  (flagged in notes)" if n in flagged else ""
        ax.text(vals[i] + 0.03, i, f"{vals[i]:.2f}{mark}", va="center", fontsize=7)
    ax.set_xlabel("mean |effect| on P(CT wins), percentage points per second")
    ax.set_title("Top 25 inputs by SHAP effect (colour = bundle)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "top25_inputs.png"), dpi=130)
    plt.close(fig)

    dep = pl.read_csv(os.path.join(tables, "dependence_top10.csv"))
    show = top[:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for ax, f in zip(axes.ravel(), show):
        d = dep.filter((pl.col("feature") == f) & pl.col("mean_value").is_not_null()).sort("mean_value")
        ax.plot(d["mean_value"].to_numpy(), d["mean_signed_effect_pp"].to_numpy(), marker="o", ms=3, color="#4C72B0")
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("input value (bin mean)", fontsize=7)
        ax.set_ylabel("effect, pp", fontsize=7)
        ax.tick_params(labelsize=7)
        if f in NOTES:
            ax.set_facecolor("#fff4e6")
    fig.suptitle("Signed effect on P(CT wins) against each input's value (binned, validation sample)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "dependence_top8.png"), dpi=120)
    plt.close(fig)

    # SHAP importance vs black-box permutation importance
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = fi["mean_abs_effect_pp"].to_numpy()
    ys = fi["perm_delta_logloss"].to_numpy()
    ax.scatter(xs, ys, s=12, color="#4C72B0")
    for f, a, b in zip(fi["feature"].to_list(), xs, ys):
        if b is not None and (a > np.percentile(xs, 90) or b > np.nanpercentile(ys, 90)):
            ax.annotate(f, (a, b), fontsize=6)
    ax.set_xlabel("TreeSHAP: mean |effect| on P(CT wins), pp")
    ax.set_ylabel("permutation: log-loss increase when shuffled")
    ax.set_title("Explainability (SHAP) vs black-box importance (permutation)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs, "shap_vs_permutation.png"), dpi=130)
    plt.close(fig)

    print("bundle table:")
    print(gi.select(["bundle", "n_inputs", "mean_abs_effect_pp", "mean_signed_effect_pp", "perm_delta_logloss"]))
    print("top 15 inputs:")
    print(fi.select(["feature", "bundle", "mean_abs_effect_pp", "mean_signed_effect_pp", "perm_delta_logloss", "rank_perm"]).head(15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "explain", "report"], required=True)
    a = ap.parse_args()
    {"train": stage_train, "explain": stage_explain, "report": stage_report}[a.stage]()


if __name__ == "__main__":
    main()
