"""CUDA versus CPU prediction parity for the 57-input XGBoost (de_mirage, frozen split, every 10th second).

The fits run on CUDA, and the prediction path has logged "Falling back to prediction using DMatrix due to mismatched devices". Check:
the same booster, predicted on CUDA (the path the fits use) and on a CPU copy of it, agrees to 1e-6 (max absolute difference) on the
test rows. Pre-registered: agreement to 1e-6 means the GPU results used throughout are not device artifacts.

Writes data/diagnostics/device_parity_check.json.
Run in the cs2gpu environment (polars, xgboost):
  C:/Users/13144/.pixi/workspace-envs/cs2gpu-15282324412032665829/envs/default/python.exe scripts/model/device_parity_check.py
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import wp_common as W  # noqa: E402
import defuse_race_eval as D  # noqa: E402

ROOT = W.ROOT
DIAG = os.path.join(ROOT, "data", "diagnostics")
OUT = os.path.join(DIAG, "device_parity_check.json")


def main():
    t0 = time.time()
    feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
    assert len(feats) == 57, len(feats)
    G = D.base_rows(feats)
    tr = G.filter(pl.col("split") == "train")
    va = G.filter(pl.col("split") == "val")
    te = G.filter(pl.col("split") == "test")
    m, best = W.fit_xgb(tr.select(feats).to_numpy().astype(np.float32), tr["label_ct_win"].to_numpy().astype(np.int64),
                        va.select(feats).to_numpy().astype(np.float32), va["label_ct_win"].to_numpy().astype(np.int64),
                        mono=W.mono_for(feats), seed=0, device="cuda")
    X = te.select(feats).to_numpy().astype(np.float32)
    rounds = best + 1
    p_cuda = m.predict_proba(X, iteration_range=(0, rounds))[:, 1]
    bst_cpu = xgb.Booster()
    bst_cpu.load_model(bytearray(m.get_booster().save_raw("ubj")))
    bst_cpu.set_param({"device": "cpu"})
    p_cpu = bst_cpu.predict(xgb.DMatrix(X), iteration_range=(0, rounds))
    diff = float(np.max(np.abs(p_cuda - p_cpu)))
    res = {"rows_test": int(len(X)), "trees_used": int(rounds), "max_abs_diff": diff, "agrees_1e6": bool(diff <= 1e-6),
           "seconds": round(time.time() - t0)}
    os.makedirs(DIAG, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
