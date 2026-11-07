"""Validation-only: fit one of the two pinned arms (A77, A77+DM) at the pinned setting, seeds given, through the sprint's own worker
functions. Derived from ablation_ll_gdt/code/fit_arm.py. The changes are the two arms, the reduced base (two armor columns dropped),
the DM margin column for a77dm, the smoke mode, and the removal of the base-only reproduction check.

Uses experiments/sprint/sprint_worker.py (prepare_set, fit_cell_prepared, cell_features) unchanged. The two arms are registered by
adding keys to the worker's SETS and GROUPS at run time; the worker file is not edited. Reads only the sprint training matrix
(G:/pureskill/csds-cs2/data/sprint/matrix_trainctr) and, for a77dm, the copy with margin appended that build_margin_matrix.py wrote
under the scratch folder. Does not open holdout B, the Oct 26-31 holdout or the frozen test. Writes results and per-row validation
predictions under the scratch folder only.

    cs2gpu python.exe -I -X utf8 -B fit_arm77.py --arm a77 --smoke
    cs2gpu python.exe -I -X utf8 -B fit_arm77.py --arm a77dm --smoke
    cs2gpu python.exe -I -X utf8 -B fit_arm77.py --arm a77 --seeds 0,1,2
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np

SPRINT = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/experiments/sprint"
SCRATCH = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/a77dm"
MATRIX = "G:/pureskill/csds-cs2/data/sprint/matrix_trainctr"
MATRIX_DM = os.path.join(SCRATCH, "matrix_dm")  # the same rows with margin as column 84, written by build_margin_matrix.py
DEPTH, ETA, MCW = 4, 0.03, 20          # pinned: the selected full setting in the sprint report
EXPECTED_FEATS = {"a77": 77, "a77dm": 78}
BASE_DROP = ("st_armor_sum_ct", "st_armor_sum_t")  # A77 drops these two of the 59 base inputs, leaving 57
EXPECTED_BASE = 57
ARMS = {"a77": ["base", "GEO", "BLIND"], "a77dm": ["base", "GEO", "BLIND", "DM"]}  # DM is the margin column
SMOKE_ROWS = 50_000  # the first fit rows only: builds the arm, reads no validation row, computes no score
EXPECTED_VAL_ROWS, EXPECTED_VAL_MATCHES = 1515228, 1168

sys.path.insert(0, SPRINT)
import polars as pl  # noqa: E402
import sprint_worker as SW  # noqa: E402  (pinned worker code, not edited)
import xgboost as xgb  # noqa: E402

SW.GROUPS["DM"] = ["margin"]
SW.SETS.update(ARMS)


def load_arm(arm):
    """The matrix for one arm: the sprint matrix, or for a77dm the same rows with margin appended. Validation keys and y come from
    the sprint matrix folder unchanged."""
    meta = json.load(open(os.path.join(MATRIX, "meta.json")))
    assert meta["pos"] == SW.POS_NAME, f"matrix built from {meta['pos']}"
    if arm == "a77dm":
        dm_meta = json.load(open(os.path.join(MATRIX_DM, "meta.json")))
        feats_all = dm_meta["feats_all"]
        assert feats_all == meta["feats_all"] + ["margin"], "margin matrix does not extend the sprint matrix by one column"
        X = np.load(os.path.join(MATRIX_DM, "X.npy"), mmap_mode="r")
        matrix_dir = MATRIX_DM
    else:
        feats_all = meta["feats_all"]
        X = np.load(os.path.join(MATRIX, "X.npy"), mmap_mode="r")
        matrix_dir = MATRIX
    assert X.shape == (meta["rows"], len(feats_all)) and X.dtype == np.float32, (X.shape, X.dtype)
    y = np.load(os.path.join(MATRIX, "y.npy"), mmap_mode="r")
    val_keys = pl.read_parquet(os.path.join(MATRIX, "val_keys.parquet"))
    base = [f for f in meta["base"] if f not in BASE_DROP]
    assert len(base) == EXPECTED_BASE, len(base)
    return meta, feats_all, base, X, y, val_keys, matrix_dir


def smoke(arm, meta, feats_all, base, X, y):
    """Build the arm on the first fit rows only: the QuantileDMatrix, the monotone tuple and two boosting rounds on CPU. No
    validation row is read and no score is computed."""
    assert SMOKE_ROWS <= meta["n_fit"]
    feats = SW.cell_features(arm, base)
    idx = [feats_all.index(f) for f in feats]
    Xs = np.ascontiguousarray(X[:SMOKE_ROWS][:, idx])
    ys = np.asarray(y[:SMOKE_ROWS])
    d = xgb.QuantileDMatrix(Xs, ys, max_bin=256)
    mono = tuple(SW.W.mono_for(feats))
    assert d.num_row() == SMOKE_ROWS and d.num_col() == len(feats) == EXPECTED_FEATS[arm], (d.num_row(), d.num_col(), len(feats))
    assert len(mono) == len(feats)
    bst = xgb.train({"objective": "binary:logistic", "eta": ETA, "max_depth": DEPTH, "min_child_weight": MCW, "subsample": 0.8,
                     "tree_method": "hist", "device": "cpu", "seed": 0, "monotone_constraints": mono}, d, num_boost_round=2)
    summary = {"arm": arm, "inputs": len(feats), "first_feature": feats[0], "last_feature": feats[-1],
               "smoke_fit_rows": SMOKE_ROWS, "dmatrix_rows": d.num_row(), "dmatrix_cols": d.num_col(),
               "monotone_entries": len(mono), "rounds_built": bst.num_boosted_rounds(),
               "smoke_slice_nan_in_last_column": int(np.isnan(Xs[:, -1]).sum()), "validation_rows_read": 0, "score": None}
    print("SMOKE " + json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(EXPECTED_FEATS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--smoke", action="store_true", help="build the arm on the first fit rows only: no validation read, no score")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    res_dir = os.path.join(SCRATCH, "out", "fits", "results")
    pred_dir = os.path.join(SCRATCH, "out", "fits", "preds")

    meta, feats_all, base, X, y, val_keys, matrix_dir = load_arm(a.arm)
    feats = SW.cell_features(a.arm, base)
    assert len(feats) == EXPECTED_FEATS[a.arm], (a.arm, len(feats))
    assert (feats[-1] == "margin") == (a.arm == "a77dm"), feats[-3:]
    if a.smoke:
        smoke(a.arm, meta, feats_all, base, X, y)
        return
    assert val_keys.height == meta["n_val"] == EXPECTED_VAL_ROWS, (val_keys.height, meta["n_val"])
    n_val_matches = val_keys["match_key"].n_unique()
    assert n_val_matches == EXPECTED_VAL_MATCHES, n_val_matches
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"arm {a.arm}: {len(feats)} inputs; matrix {X.shape} ({matrix_dir}); fit {meta['n_fit']:,}, early stop {meta['n_es']:,}, "
          f"validation {meta['n_val']:,} rows / {n_val_matches:,} matches; rss {SW.rss_mb()} MB", flush=True)

    data = {"X": X, "y": y, "feats_all": feats_all, "base": base, "n_fit": meta["n_fit"], "n_es": meta["n_es"],
            "val_keys": val_keys}
    t_arm = time.time()
    prep = SW.prepare_set(data, a.arm)
    for seed in seeds:
        cell = {"id": f"{a.arm}_d{DEPTH}_e{ETA}_m{MCW}_s{seed}", "set": a.arm, "depth": DEPTH, "eta": ETA, "mcw": MCW, "seed": seed}
        t0 = time.time()
        r = SW.fit_cell_prepared(prep, cell, "cuda", pred_dir, data)
        r.update(cell)
        r.update({"status": "ok", "device": "cuda", "inputs": len(feats), "feature_names": feats,
                  "wall_seconds": round(time.time() - t0, 1), "matrix": matrix_dir,
                  "pinned": {"depth": DEPTH, "eta": ETA, "min_child_weight": MCW, "subsample": 0.8,
                             "tree_method": "hist", "objective": "binary:logistic", "early_stopping_rounds": 100,
                             "num_boost_round_cap": 3000, "max_bin": 256}})
        with open(os.path.join(res_dir, cell["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"{cell['id']}: val_logloss {r['val_logloss']:.6f} best {r['best_iteration']} fit {r['fit_seconds']}s "
              f"wall {r['wall_seconds']}s n_val {r['n_validation']:,}", flush=True)
    del prep
    gc.collect()
    print(f"arm {a.arm} done in {time.time() - t_arm:.0f}s", flush=True)


if __name__ == "__main__":
    main()
