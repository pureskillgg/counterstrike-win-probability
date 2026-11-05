"""Explainability for the 57-input submission model (winprob_v6_drop_heavy), on the same validation seconds as the 83-input run.

No retraining: loads models/winprob_v6_drop_heavy.json. Its 57 inputs are the 83-matrix columns named in its meta (no position
inputs; heavy weapons dropped). Uses the same SHAP sample rows as the 83-input run (WORK/shap_sample.npz), so the two models are
compared on identical seconds. Permutation importance uses all validation rows, as in the 83-input run.

Outputs under docs/explainability/xgb57/tables: feature_effects_57.csv, bundle_effects_57.csv, perm_importance_57.csv,
compare_83_vs_57_bundles.csv, explain_meta_57.json.

Run:  python scripts/explain/xgb57_explain.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import DOCS, GROUPS, MDIR, NOTES, WORK, logloss, sig  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MODEL = os.path.join(REPO, "models", "winprob_v6_drop_heavy.json")
META = os.path.join(REPO, "models", "winprob_v6_drop_heavy.meta.json")
OUT = os.path.join(DOCS, "xgb57", "tables")
RNG_SEED = 20261105
REPEATS_GROUP = 2
REPEATS_FEATURE = 1


def main():
    os.makedirs(OUT, exist_ok=True)
    vm = json.load(open(META, encoding="utf-8"))
    feats57 = vm["features"]
    assert len(feats57) == 57 and len(set(feats57)) == 57, "expected the 57 inputs of the v6 model"
    mm = json.load(open(os.path.join(MDIR, "meta.json")))
    feats83 = mm["feats_all"]
    missing = [f for f in feats57 if f not in feats83]
    assert not missing, f"v6 inputs not in the matrix: {missing}"
    cols = [feats83.index(f) for f in feats57]
    a2 = mm["n_fit"] + mm["n_es"]
    X = np.load(os.path.join(MDIR, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(MDIR, "y.npy"), mmap_mode="r")
    Xv = np.asarray(X[a2:][:, cols], dtype=np.float32)
    yv = np.asarray(y[a2:], dtype=np.float64)
    n = Xv.shape[0]
    print(f"model {vm['version']}: {len(feats57)} inputs, best iteration {vm['best_iteration']}, "
          f"input file {vm.get('input_file')}; validation rows {n:,}", flush=True)

    bst = xgb.Booster()
    bst.load_model(MODEL)
    iters = (0, int(vm["best_iteration"]) + 1)

    # identical SHAP sample to the 83-input run
    S = np.load(os.path.join(WORK, "shap_sample.npz"))
    idx = S["idx"]
    dS = xgb.DMatrix(Xv[idx])
    t0 = time.time()
    contrib = bst.predict(dS, pred_contribs=True, iteration_range=iters)
    z = bst.predict(dS, output_margin=True, iteration_range=iters)
    add_err = float(np.abs(contrib.sum(axis=1) - z).max())
    assert add_err < 1e-3, f"SHAP additivity error {add_err}"
    phi = contrib[:, :57].astype(np.float64)
    p = sig(z.astype(np.float64))
    print(f"TreeSHAP on {len(idx):,} rows in {time.time() - t0:.0f}s, additivity max error {add_err:.2e}", flush=True)

    dV = xgb.DMatrix(Xv)
    base_p = bst.predict(dV, iteration_range=iters)
    base_ll = logloss(base_p, yv)
    print(f"validation log loss, all {n:,} rows: {base_ll:.6f}", flush=True)

    grp_of = {f: g for g, cols_ in GROUPS.items() for f in cols_ if f in feats57}
    bundles = {g: [f for f in cols_ if f in feats57] for g, cols_ in GROUPS.items()}
    bundles = {g: v for g, v in bundles.items() if v}

    # ---- SHAP effects ----
    feat_rows = []
    for j, f in enumerate(feats57):
        pp = 100.0 * (p - sig(z - phi[:, j]))
        feat_rows.append({"feature": f, "bundle": grp_of.get(f, ""), "note": NOTES.get(f, ""),
                          "mean_abs_shap_logit": float(np.abs(phi[:, j]).mean()),
                          "mean_abs_effect_pp": float(np.abs(pp).mean()),
                          "mean_signed_effect_pp": float(pp.mean())})
    bundle_rows = []
    for g, members in bundles.items():
        idc = [feats57.index(f) for f in members]
        phi_g = phi[:, idc].sum(axis=1)
        pp_g = 100.0 * (p - sig(z - phi_g))
        bundle_rows.append({"bundle": g, "n_inputs": len(members), "mean_abs_bundle_shap_logit": float(np.abs(phi_g).mean()),
                            "mean_abs_effect_pp": float(np.abs(pp_g).mean()), "mean_signed_effect_pp": float(pp_g.mean()),
                            "inputs": ", ".join(members)})
    for name, members in [("ALL_LOADOUT", [f for g in bundles if g.startswith("LOADOUT_") for f in bundles[g]])]:
        idc = [feats57.index(f) for f in members]
        phi_g = phi[:, idc].sum(axis=1)
        pp_g = 100.0 * (p - sig(z - phi_g))
        bundle_rows.append({"bundle": name, "n_inputs": len(members), "mean_abs_bundle_shap_logit": float(np.abs(phi_g).mean()),
                            "mean_abs_effect_pp": float(np.abs(pp_g).mean()), "mean_signed_effect_pp": float(pp_g.mean()),
                            "inputs": ", ".join(members)})

    # ---- permutation (black box) on all validation rows ----
    rng = np.random.default_rng(RNG_SEED)
    jobs = [("bundle", g, [feats57.index(f) for f in m], REPEATS_GROUP) for g, m in bundles.items()]
    jobs += [("feature", f, [feats57.index(f)], REPEATS_FEATURE) for f in feats57]
    perm_rows = []
    t0 = time.time()
    for kind, name, cc, reps in jobs:
        deltas = []
        for _ in range(reps):
            perm = rng.permutation(n)
            Xp = Xv.copy()
            Xp[:, cc] = Xv[perm][:, cc]
            pr = bst.predict(xgb.DMatrix(Xp), iteration_range=iters)
            deltas.append(logloss(pr, yv) - base_ll)
            del Xp
        perm_rows.append({"kind": kind, "name": name, "perm_delta_logloss_mean": float(np.mean(deltas)),
                          "repeats": reps})
    print(f"permutation: {len(jobs)} jobs in {time.time() - t0:.0f}s", flush=True)
    perm = pl.DataFrame(perm_rows)
    perm.write_csv(os.path.join(OUT, "perm_importance_57.csv"))
    perm_b = {r["name"]: r["perm_delta_logloss_mean"] for r in perm.filter(pl.col("kind") == "bundle").iter_rows(named=True)}
    perm_f = {r["name"]: r["perm_delta_logloss_mean"] for r in perm.filter(pl.col("kind") == "feature").iter_rows(named=True)}

    feats_df = pl.DataFrame(feat_rows).with_columns(
        pl.col("feature").map_elements(lambda f: perm_f.get(f), return_dtype=pl.Float64).alias("perm_delta_logloss")
    ).sort("mean_abs_effect_pp", descending=True)
    feats_df.write_csv(os.path.join(OUT, "feature_effects_57.csv"))

    bundles_df = pl.DataFrame(bundle_rows).with_columns(
        pl.col("bundle").map_elements(lambda b: perm_b.get(b), return_dtype=pl.Float64).alias("perm_delta_logloss")
    ).sort("mean_abs_effect_pp", descending=True)
    bundles_df.write_csv(os.path.join(OUT, "bundle_effects_57.csv"))

    # ---- compare with the 83-input bundles on the same seconds ----
    b83 = pl.read_csv(os.path.join(DOCS, "tables", "bundle_importance_and_effects.csv")).select(
        ["bundle", pl.col("mean_abs_effect_pp").alias("effect_pp_83"), pl.col("perm_delta_logloss").alias("perm_83")])
    b57 = bundles_df.select(["bundle", pl.col("mean_abs_effect_pp").alias("effect_pp_57"),
                             pl.col("perm_delta_logloss").alias("perm_57")])
    cmp_df = b83.join(b57, on="bundle", how="full", coalesce=True).sort("effect_pp_83", descending=True, nulls_last=True)
    cmp_df.write_csv(os.path.join(OUT, "compare_83_vs_57_bundles.csv"))

    json.dump({"model": vm["version"], "n_inputs": 57, "best_iteration": int(vm["best_iteration"]),
               "input_file": vm.get("input_file"), "validation_logloss_all_rows": base_ll, "validation_rows": int(n),
               "shap_rows": int(len(idx)), "shap_additivity_max_error": add_err, "repeats_group": REPEATS_GROUP,
               "repeats_feature": REPEATS_FEATURE, "note": "no position inputs; heavy weapons dropped"},
              open(os.path.join(OUT, "explain_meta_57.json"), "w", encoding="utf-8"), indent=1)
    print("bundles (57):")
    print(bundles_df.select(["bundle", "n_inputs", "mean_abs_effect_pp", "mean_signed_effect_pp", "perm_delta_logloss"])
          .to_pandas().to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("comparison 83 vs 57:")
    print(cmp_df.to_pandas().to_string(index=False, float_format=lambda v: f"{v:.5g}"))


if __name__ == "__main__":
    main()
