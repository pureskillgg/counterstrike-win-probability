"""Learning curve, XGBoost arm: the 57-input recipe (W.fit_xgb, as arm A in sitecheck_xgb.py) trained on a fraction of the train matches.
Pre-registered: Meridian pin dc66a027 (27 Oct 2026). Validation only: the test set is not read here.

Subsets: nested draws by match key. The train match keys are sorted, permuted with numpy default_rng(SUBSET_SEED), and the first
round(fraction * n) keys are kept. The GRU learning-curve runs (train_temporal_allmaps.py --match-fraction) use the same rule.
Early stopping on the validation split, as in arm A. Seed 0 only.

Usage (CPU; the GPU is busy with the GRU runs):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe -I -X utf8 -u scripts/model/learning_curve_xgb.py --fraction 0.25
Writes data/diagnostics/learning_curve/xgb_lc<pct>_metrics.json and xgb_lc<pct>_val_preds.parquet.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wp_common as W  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUTDIR = os.path.join(DIAG, "learning_curve")
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
KEY = ["match_key", "round", "second"]
SUBSET_SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not 0.0 < a.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    pct = int(round(a.fraction * 100))

    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    geo_old = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo", f"{m}.parquet")) for m in MAPS])
    geo_new = pl.concat([pl.read_parquet(os.path.join(DIAG, "allmaps_geo_trainsites", f"{m}.parquet")) for m in MAPS])
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = W.drop_roster_defect_rounds(C)
    C = C.filter(pl.col("label_ct_win").is_not_null()).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    man = W.load_manifest(os.path.join(ROOT, "data", "combined", "split_manifest.json"))["assign"]
    A = pl.DataFrame({"match_key": list(man.keys()), "split": list(man.values())})
    # the same population as sitecheck_xgb.py (and t0_logreg_vs_xgb.py)
    pop = (C.select(KEY + ["label_ct_win"])
             .join(geo_old.select(KEY).unique(KEY), on=KEY, how="semi")
             .join(geo_new.select(KEY).unique(KEY), on=KEY, how="semi")
             .join(A, on="match_key", how="inner"))
    pop = pop.unique(KEY, keep="first", maintain_order=True).sort(KEY)
    G = pop.join(C.select(KEY + feats), on=KEY, how="left", maintain_order="left")
    del C, pop, geo_old, geo_new

    # string matching stays in polars: numpy on 30 million string keys is far too slow
    tr_pl = G["split"] == "train"
    is_va = (G["split"] == "val").to_numpy()
    ukeys = np.sort(G.filter(tr_pl)["match_key"].unique().to_numpy())
    n_train_matches = int(ukeys.size)
    if a.fraction < 1.0:
        perm = np.random.default_rng(SUBSET_SEED).permutation(ukeys)
        keep = perm[:max(1, int(round(a.fraction * ukeys.size)))]
        is_tr = (tr_pl & G["match_key"].is_in(keep.tolist())).to_numpy()
        n_used = int(keep.size)
    else:
        is_tr = tr_pl.to_numpy()
        n_used = n_train_matches
    n_tr, n_va = int(is_tr.sum()), int(is_va.sum())
    print(f"fraction {a.fraction}: {n_used:,} of {n_train_matches:,} train matches, train rows {n_tr:,}, validation rows {n_va:,} "
          f"({time.time() - t0:.0f} s)", flush=True)

    Xtr = np.empty((n_tr, len(feats)), dtype=np.float32)
    Xva = np.empty((n_va, len(feats)), dtype=np.float32)
    for j, c in enumerate(feats):
        col = G[c].to_numpy()
        Xtr[:, j] = col[is_tr]
        Xva[:, j] = col[is_va]
        del col
    ytr = G["label_ct_win"].to_numpy()[is_tr].astype(np.int64)
    yva = G["label_ct_win"].to_numpy()[is_va].astype(np.int64)
    base = (G.filter(pl.col("split") == "val").select(KEY + ["label_ct_win"])
              .rename({"label_ct_win": "label"}).with_columns(pl.col("label").cast(pl.Int64)))
    del G
    print(f"matrices built ({time.time() - t0:.0f} s); fitting", flush=True)

    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono=W.mono_for(feats), seed=0, device=a.device)
    p_va = W.predict(m, best, Xva)
    ll = float(W.ll_vec(p_va, yva).mean())
    print(f"best iteration {best}; validation log loss {ll:.6f} ({time.time() - t0:.0f} s)", flush=True)

    base.with_columns(pl.Series("p", p_va.astype(np.float64))).write_parquet(os.path.join(OUTDIR, f"xgb_lc{pct}_val_preds.parquet"))
    res = {"model": "xgb_arm_A", "fraction": a.fraction, "subset_seed": SUBSET_SEED, "train_matches_used": n_used,
           "train_matches_available": n_train_matches, "train_rows": n_tr, "validation_rows": n_va, "best_iteration": int(best),
           "validation_log_loss": ll, "device": a.device, "seconds": round(time.time() - t0)}
    W.dump_json(res, os.path.join(OUTDIR, f"xgb_lc{pct}_metrics.json"))
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
