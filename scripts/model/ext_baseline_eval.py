"""Does the extra training data help the 57-input XGBoost? (paired, GPU)

Two arms, same features, same frozen split (data/combined/split_manifest.json). Both are scored on the SAME test rows: the frozen
de_mirage test matches present in both caches.
  O  trained on the train matches of the original snapshot cache (data/combined/cache_snapshot_2026.parquet)
  E  trained on the train matches of the extended cache (data/combined/cache_snapshot_2026_ext.parquet): the same matches plus the
     2026-10-01..21 additions, which the manifest assigns to train only. Test and validation matches are unchanged.

Scope (--scope):
  all        train on every map, early-stop on every map's validation matches, score the de_mirage test matches. This is the reference
             setup: the 0.4512 model is trained on all maps.
  de_mirage  train and early-stop on de_mirage only (a single-map comparison, kept for reference).

Pre-registered (set before the all-maps run was read):
  the extra data helps if mean(E - O) <= -0.0005 and the 95% upper bound < 0.
  arm O reproduces the reference if, under scope all, its de_mirage test log loss is within 0.001 of 0.4512.

  ...cs2gpu python.exe -I -X utf8 -u scripts\\model\\ext_baseline_eval.py --scope all
Writes data/diagnostics/ext_baseline_<scope>.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
KEY = ["match_key", "round", "second"]
CACHES = {"O": "cache_snapshot_2026.parquet", "E": "cache_snapshot_2026_ext.parquet"}
N_BOOT = 1000
STRIDE = 2
MAP = "de_mirage"


def load(cache, feats, man, scope):
    """The rows the arms need: train seconds at STRIDE, validation seconds, and the frozen de_mirage test seconds. The filters are
    pushed into a lazy scan and the features are cast to float32 before collecting, because the extended cache is 31M rows."""
    in_scope = pl.lit(True) if scope == "all" else (pl.col("map_name") == MAP)
    keep = (((pl.col("split") == "train") & (pl.col("second") % STRIDE == 0) & in_scope)
            | ((pl.col("split") == "val") & in_scope)
            | ((pl.col("split") == "test") & (pl.col("map_name") == MAP)))
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())}).lazy()
    G = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", cache))
         .select(KEY + ["label_ct_win", "map_name"] + feats)
         .filter(pl.col("label_ct_win").is_not_null())
         .join(A, on="match_key", how="inner")
         .filter(keep)
         .with_columns([pl.col(f).cast(pl.Float32) for f in feats])
         .collect())
    G = W.drop_roster_defect_rounds(G)
    return G.sort(KEY)


def paired(match, y, pa, pb, seed=0):
    la = W.ll_vec(pa, y)
    lb = W.ll_vec(pb, y)
    uniq, inv = np.unique(match, return_inverse=True)
    M = len(uniq)
    cnt = np.bincount(inv, minlength=M).astype(float)
    sa = np.bincount(inv, weights=la, minlength=M)
    sb = np.bincount(inv, weights=lb, minlength=M)
    rng = np.random.default_rng(seed)
    d = np.empty(N_BOOT)
    for b in range(N_BOOT):
        w = np.bincount(rng.integers(0, M, M), minlength=M).astype(float)
        d[b] = ((w * sa).sum() - (w * sb).sum()) / (w * cnt).sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean_a": float(la.mean()), "mean_b": float(lb.mean()), "diff": float(la.mean() - lb.mean()),
            "ci95": [float(lo), float(hi)], "rows": int(len(y)), "matches": int(M)}


def fit(G, cols, name, seed):
    tr = G.filter((pl.col("split") == "train") & (pl.col("second") % STRIDE == 0))
    va = G.filter(pl.col("split") == "val")
    m, best = W.fit_xgb(tr.select(cols).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(cols).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(cols), seed=seed, device="cuda")
    print(f"  {name}: best iteration {best}, train rows {tr.height:,}, validation rows {va.height:,}", flush=True)
    return m, best, tr.height, va.height


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["all", "de_mirage"], default="all")
    ap.add_argument("--seed", type=int, default=0, help="XGBoost seed for both arms (row subsampling and column draws)")
    a = ap.parse_args()
    scope, seed = a.scope, a.seed
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    G = {arm: load(cache, feats, man, scope) for arm, cache in CACHES.items()}

    # de_mirage matches per manifest split, and how many of them each cache holds (reconciles the manifest with the arrays)
    dm = set(pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "map_name"])
              .filter(pl.col("map_name") == MAP)["match_key"].to_list())
    manifest_dm = {s: sorted(k for k, v in man.items() if v == s and k in dm) for s in ["train", "val", "test"]}
    in_cache = {arm: {s: sorted(G[arm].filter((pl.col("split") == s) & (pl.col("map_name") == MAP))["match_key"].unique().to_list())
                      for s in ["train", "val", "test"]} for arm in CACHES}
    missing = {arm: {s: sorted(set(manifest_dm[s]) - set(in_cache[arm][s])) for s in ["train", "val", "test"]} for arm in CACHES}
    print({s: len(v) for s, v in manifest_dm.items()}, flush=True)
    for arm in CACHES:
        print(f"  {arm}: matches in cache per split", {s: len(v) for s, v in in_cache[arm].items()},
              "| manifest matches missing from cache", {s: len(v) for s, v in missing[arm].items()}, flush=True)

    test_keys = sorted(set(in_cache["O"]["test"]) & set(in_cache["E"]["test"]))
    models, best, rows_train, rows_val = {}, {}, {}, {}
    for arm in CACHES:
        models[arm], best[arm], rows_train[arm], rows_val[arm] = fit(G[arm], feats, arm, seed)
    te = {arm: G[arm].filter((pl.col("split") == "test") & pl.col("match_key").is_in(test_keys)).sort(KEY) for arm in CACHES}
    assert te["O"].height == te["E"].height and (te["O"].select(KEY).to_numpy() == te["E"].select(KEY).to_numpy()).all(), "test rows differ"
    p = {arm: W.predict(models[arm], best[arm], te[arm].select(feats).to_numpy().astype(np.float32)) for arm in CACHES}
    y = te["O"]["label_ct_win"].to_numpy().astype(np.int64)
    match = te["O"]["match_key"].to_numpy()
    suffix = "" if seed == 0 else f"_seed{seed}"
    # per-row predictions of both arms on the same test rows, so other models can be paired with them later
    pl.DataFrame({"match_key": match, "round": te["O"]["round"].to_numpy(), "second": te["O"]["second"].to_numpy(),
                  "label": y, "p_O": p["O"], "p_E": p["E"]}).write_parquet(os.path.join(DIAG, f"ext_baseline_{scope}{suffix}_preds.parquet"))
    r = paired(match, y, p["E"], p["O"])
    res = {
        "scope": scope,
        "seed": seed,
        "rows_test": int(te["O"].height),
        "matches_test": len(test_keys),
        "manifest_matches_de_mirage": {s: len(v) for s, v in manifest_dm.items()},
        "missing_from_cache": {arm: {s: len(v) for s, v in missing[arm].items()} for arm in CACHES},
        "rows_train_stride2": rows_train,
        "rows_validation": rows_val,
        "best_iteration": best,
        "logloss_test": {arm: float(W.ll_vec(p[arm], y).mean()) for arm in CACHES},
        "E_minus_O": r,
        "E_helps": bool(r["diff"] <= -0.0005 and r["ci95"][1] < 0),
        "threshold": "extra data helps if mean(E-O) <= -0.0005 and CI upper < 0",
        "seconds": round(time.time() - t0),
    }
    os.makedirs(DIAG, exist_ok=True)
    name = f"ext_baseline_{scope}.json" if seed == 0 else f"ext_baseline_{scope}_seed{seed}.json"
    with open(os.path.join(DIAG, name), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
