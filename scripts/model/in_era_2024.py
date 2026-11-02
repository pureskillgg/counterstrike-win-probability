"""2024 trained on 2024: how learnable is 2024, and does a 2024-trained model beat the 2026-trained model on 2024 test matches?

Rows: data/combined/cache_snapshot_2024.parquet (the builder's roster-defect and inconsistent rounds already removed), minus the four
matches with no status file, as in eval_2024_frozen.py. Split by match, seed 0: 70% train, 10% validation, 20% test.

Arms (frozen recipe, W.fit_xgb, 57 inputs):
  R2024  trained on 2024 train matches, early-stopped on 2024 validation matches
  F2026  the frozen 2026 model (no refit), scored on the same 2024 test matches
Paired on the 2024 test seconds (spatial_feature_eval.paired, 1,000 resamples).

Pre-registered (written before any result was read): the 2024-trained model beats the frozen 2026 model on 2024 test matches if the mean
paired difference (R2024 minus F2026) is <= -0.005 and the CI upper bound is < 0.

Run in the cs2gpu environment (polars, xgboost, CUDA):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/in_era_2024.py
Writes data/diagnostics/in_era_2024.json.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
from spatial_feature_eval import paired  # noqa: E402

ROOT = W.ROOT
OUT = os.path.join(ROOT, "data", "diagnostics", "in_era_2024.json")
KEY = ["match_key", "round", "second"]


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    C = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2024.parquet"),
                        columns=sorted(set(KEY + ["match_key", "label_ct_win", "map_name"] + feats)))
    C = C.filter(pl.col("label_ct_win").is_not_null())
    status = set()
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "status_1hz_2024", "*.parquet"))):
        status.update(pl.read_parquet(f, columns=["match_key"])["match_key"].unique().to_list())
    C = C.filter(pl.col("match_key").is_in(list(status))).with_columns(pl.col("round").cast(pl.Int64), pl.col("second").cast(pl.Int64))
    matches = sorted(C["match_key"].unique().to_list())
    rng = np.random.default_rng(0)
    order = [matches[i] for i in rng.permutation(len(matches))]
    n = len(order)
    tr_m, va_m, te_m = set(order[:int(0.7 * n)]), set(order[int(0.7 * n):int(0.8 * n)]), set(order[int(0.8 * n):])
    tr = C.filter(pl.col("match_key").is_in(list(tr_m)))
    va = C.filter(pl.col("match_key").is_in(list(va_m)))
    te = C.filter(pl.col("match_key").is_in(list(te_m)))
    print(f"matches train {len(tr_m)}, val {len(va_m)}, test {len(te_m)}; rows train {tr.height:,}, val {va.height:,}, test {te.height:,} "
          f"({time.time() - t0:.0f} s)", flush=True)

    m, best = W.fit_xgb(tr.select(feats).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(feats).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(feats), seed=0, device="cuda")
    p_new = W.predict(m, best, te.select(feats).to_numpy().astype(np.float32))
    booster = xgb.Booster()
    booster.load_model(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.json"))
    p_old = booster.predict(xgb.DMatrix(te.select(feats).to_numpy().astype(np.float32), feature_names=feats))
    y = te["label_ct_win"].to_numpy().astype(np.int64)
    match = te["match_key"].to_numpy()
    d = paired(match, y, p_new, p_old)
    res = {"matches": {"train": len(tr_m), "val": len(va_m), "test": len(te_m)},
           "rows": {"train": tr.height, "val": va.height, "test": te.height},
           "best_iteration": int(best), "log_loss_R2024": float(W.ll_vec(p_new, y).mean()),
           "log_loss_F2026": float(W.ll_vec(p_old, y).mean()), "R2024_minus_F2026": d,
           "passes": bool(d["diff"] <= -0.005 and d["ci95"][1] < 0), "seconds": round(time.time() - t0)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ["log_loss_R2024", "log_loss_F2026", "R2024_minus_F2026", "passes"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
