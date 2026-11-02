"""S4: retrain reference A and arm SN (XGBoost) on the FROZEN training keys, CPU, for the seeds given on the command line, and
compare the VALIDATION predictions with the saved ones for the same seed.

Recipe = sitecheck_xgb.py: W.fit_xgb (n_estimators 3000, eta 0.05, depth 4, min_child_weight 20, subsample 0.8, hist, early stopping
100 on validation logloss), monotone constraints from W.mono_for. A = the 57 frozen inputs; SN = 57 + st0..st3 from the train-only
centres (allmaps_geo_trainsites). Differences from the original run: device=cpu (GPU busy), the split is the frozen backup manifest
(not data/combined/split_manifest.json), and the de_mirage TEST seconds are not scored. Row order is the one sitecheck_xgb.py uses
(sorted by match_key, round, second), because XGBoost row subsampling depends on it.
Rule (brief S4): each model needs max |p - saved p| < 1e-4 on validation rows; the first failing model stops the run (exit 3).
Writes data/holdout_b/s4_xgb_retrain.json after every model.
Usage: <cs2 python> -I -X utf8 -u scripts/model/holdout_b/s4_xgb_retrain.py A:0 SN:0 A:1 ...
"""
import gc
import json
import os
import sys
import time

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
import wp_common as W  # noqa: E402

t0 = time.time()
KEY = ["match_key", "round", "second"]
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
CACHE = os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet")
BK = os.path.join(ROOT, "data", "combined", "_backup_pre_22_26_20261027", "split_manifest.json")
OUT = os.path.join(ROOT, "data", "holdout_b", "s4_xgb_retrain.json")
GEO_OLD = [os.path.join(ROOT, "data", "diagnostics", "allmaps_geo", f"{m}.parquet") for m in MAPS]
GEO_NEW = [os.path.join(ROOT, "data", "diagnostics", "allmaps_geo_trainsites", f"{m}.parquet") for m in MAPS]
feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
assert len(feats) == 57
NST = [f"nst{i}" for i in range(4)]
H = set(open(os.path.join(ROOT, "data", "holdout_b", "holdout_keys.txt"), encoding="utf-8").read().split())
man = W.load_manifest(BK)["assign"]
tv = {k: (0 if v == "train" else 1) for k, v in man.items() if v in ("train", "val")}
assert not (set(tv) & H), "holdout key in the frozen train/val split"
print(f"frozen train {sum(1 for v in tv.values() if v == 0):,} matches, val {sum(1 for v in tv.values() if v == 1):,} matches", flush=True)

# (1) survivor keys: cache rows with a label, outside roster-defect rounds, with both geometry rows (inner joins, as sitecheck_xgb.py)
bad = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet")).filter(pl.col("missing_side").is_not_null())
       .select("match_key", pl.col("round").cast(pl.Int64)))
tv_list = list(tv.keys())
cache_k = (pl.scan_parquet(CACHE).select(KEY + ["label_ct_win"])
           .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
           .filter(pl.col("label_ct_win").is_not_null() & pl.col("match_key").is_in(tv_list))
           .join(bad, on=["match_key", "round"], how="anti"))
geo_old_k = pl.scan_parquet(GEO_OLD).select(KEY[0], pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
geo_new_k = pl.scan_parquet(GEO_NEW).select(KEY[0], pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
K = (cache_k.join(geo_old_k, on=KEY, how="inner").join(geo_new_k, on=KEY, how="inner")
     .unique(KEY, keep="first", maintain_order=True).sort(KEY)
     .with_columns(pl.col("label_ct_win").cast(pl.Int64).alias("label"),
                   pl.col("match_key").replace_strict(tv, default=None, return_dtype=pl.Int64).alias("split"))
     .select(KEY + ["label", "split"]).collect())
print(f"survivor rows {K.height:,} (train {int((K['split'] == 0).sum()):,}, val {int((K['split'] == 1).sum()):,}) ({time.time()-t0:.0f} s)", flush=True)
assert K.filter(pl.col("match_key").is_in(list(H))).height == 0
assert K.height == K.unique(KEY, keep="first").height

# feature sources (one row per key is asserted through the row-count check in build())
cache_f = (pl.scan_parquet(CACHE).select(KEY + feats)
           .with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64),
                         *[pl.col(f).cast(pl.Float32) for f in feats])
           .filter(pl.col("match_key").is_in(tv_list)))
geo_old_f = pl.scan_parquet(GEO_OLD).select(KEY[0], pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64),
                                             *[pl.col(f"st{i}").cast(pl.Float32) for i in range(4)])
geo_new_f = pl.scan_parquet(GEO_NEW).select(KEY[0], pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64),
                                             *[pl.col(f"st{i}").cast(pl.Float32).alias(f"nst{i}") for i in range(4)])

is_tr = (K["split"] == 0).to_numpy()
is_va = ~is_tr
y_all = K["label"].to_numpy().astype(np.int64)
Kv = K.filter(pl.col("split") == 1)
mk_va, rd_va, sc_va = Kv["match_key"].to_numpy(), Kv["round"].to_numpy(), Kv["second"].to_numpy()
y_tr = y_all[is_tr]
y_va = y_all[is_va]
CH = 2_500_000


def build(cols, geo_new):
    """Feature matrices for the train and validation rows, in K's row order, built in chunks of keys."""
    ntot = len(cols)
    n = K.height
    X_tr = np.empty((int(is_tr.sum()), ntot), dtype=np.float32)
    X_va = np.empty((int(is_va.sum()), ntot), dtype=np.float32)
    s_tr = s_va = 0
    for c0 in range(0, n, CH):
        c1 = min(n, c0 + CH)
        kc = K.slice(c0, c1 - c0).select(KEY)
        fr = kc.lazy().join(cache_f, on=KEY, how="left", maintain_order="left").join(geo_old_f, on=KEY, how="left", maintain_order="left")
        if geo_new:
            fr = fr.join(geo_new_f, on=KEY, how="left", maintain_order="left")
        F = fr.select(cols).collect()
        assert F.height == c1 - c0, "join changed the row count"
        A = F.to_numpy().astype(np.float32)
        assert not np.isnan(A).any(), "missing feature value"
        m = is_tr[c0:c1]
        ntr = int(m.sum())
        X_tr[s_tr:s_tr + ntr] = A[m]
        X_va[s_va:s_va + (A.shape[0] - ntr)] = A[~m]
        s_tr += ntr
        s_va += A.shape[0] - ntr
        del F, A, fr
    assert s_tr == X_tr.shape[0] and s_va == X_va.shape[0]
    return X_tr, X_va


res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {"models": {}}
stop = False
for spec in sys.argv[1:]:
    arm, seed = spec.split(":")
    seed = int(seed)
    cols = feats if arm == "A" else feats + NST
    print(f"== {arm} seed {seed}: building {len(cols)} inputs ({time.time()-t0:.0f} s)", flush=True)
    X_tr, X_va = build(cols, geo_new=(arm == "SN"))
    print(f"   train {X_tr.shape}, validation {X_va.shape} ({time.time()-t0:.0f} s)", flush=True)
    m, best = W.fit_xgb(X_tr, y_tr, X_va, y_va, mono=W.mono_for(cols), seed=seed, device="cpu")
    p_va = W.predict(m, best, X_va).astype(np.float64)
    ll = float(W.ll_vec(p_va, y_va).mean())
    mine = pl.DataFrame({"match_key": mk_va, "round": rd_va, "second": sc_va, "label_new": y_va, "p_new": p_va})
    sv = (pl.read_parquet(os.path.join(ROOT, "data", "diagnostics", "preds", f"allmaps_xgb_{arm}_seed{seed}.parquet"))
          .filter(pl.col("split") == "val").select("match_key", "round", "second", "label", pl.col("p").alias("p_saved")))
    j = mine.join(sv, on=["match_key", "round", "second"], how="full", coalesce=True)
    both = j.filter(pl.col("p_new").is_not_null() & pl.col("p_saved").is_not_null())
    d = np.abs(both["p_new"].to_numpy() - both["p_saved"].to_numpy())
    saved_ll = float(W.ll_vec(sv["p_saved"].to_numpy(), sv["label"].to_numpy()).mean())
    rec = {"arm": arm, "seed": seed, "best_iteration": int(best), "device": "cpu", "train_rows": int(X_tr.shape[0]),
           "val_rows_new": int(p_va.size), "val_rows_saved": int(sv.height), "val_rows_matched": int(both.height),
           "val_label_mismatch": int((both["label_new"] != both["label"]).sum()),
           "val_logloss_new": ll, "val_logloss_saved": saved_ll, "abs_diff_logloss": abs(ll - saved_ll),
           "max_abs_diff_p": float(d.max()), "mean_abs_diff_p": float(d.mean()),
           "pass_1e-4": bool(d.max() < 1e-4), "seconds": round(time.time() - t0)}
    res["models"][f"{arm}_seed{seed}"] = rec
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec), flush=True)
    del m, X_tr, X_va, mine, sv, j, both
    gc.collect()
    if not rec["pass_1e-4"]:
        print("FAIL: max |p diff| >= 1e-4; stopping (brief S4 rule)", flush=True)
        stop = True
        break
print("done", round(time.time() - t0), "s; stopped" if stop else "; all requested models pass", flush=True)
sys.exit(3 if stop else 0)
