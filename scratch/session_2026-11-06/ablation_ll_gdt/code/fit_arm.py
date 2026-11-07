"""Validation-only ablation: fit one feature arm at the pinned setting, seeds given, through the sprint's own worker functions.

Uses experiments/sprint/sprint_worker.py (prepare_set, fit_cell_prepared, cell_features, load_matrix) unchanged. Reads only the
sprint training matrix (G:/pureskill/csds-cs2/data/sprint/matrix_trainctr). Does not open holdout B, the Oct 26-31 holdout or the
frozen test. Writes results and per-row validation predictions under the scratch folder only.

    cs2gpu python.exe -I -X utf8 -B fit_arm.py --arm base --seeds 0
"""
import argparse
import gc
import json
import os
import sys
import time

SPRINT = "C:/Users/13144/Documents/pureskill/counterstrike-win-probability/experiments/sprint"
SCRATCH = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ablation_ll_gdt"
MATRIX = "G:/pureskill/csds-cs2/data/sprint/matrix_trainctr"
PINNED_VAL_LL_GPU_BASE_S0 = 0.45381  # docs/sprint/validation_gpu_fast/report.md, reproduction table, base, cuda
REPRO_TOL = 0.0001
DEPTH, ETA, MCW = 4, 0.03, 20          # pinned: the selected full setting in the sprint report
EXPECTED_FEATS = {"base": 59, "base+GEO": 75, "full": 83}
EXPECTED_VAL_ROWS, EXPECTED_VAL_MATCHES = 1515228, 1168

sys.path.insert(0, SPRINT)
import polars as pl  # noqa: E402
import sprint_worker as SW  # noqa: E402  (pinned worker code, not edited)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(EXPECTED_FEATS))
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    res_dir = os.path.join(SCRATCH, "out", "fits", "results")
    pred_dir = os.path.join(SCRATCH, "out", "fits", "preds")

    meta, X, y, val_keys = SW.load_matrix(MATRIX)
    assert meta["pos"] == SW.POS_NAME, f"matrix built from {meta['pos']}"
    assert val_keys.height == meta["n_val"] == EXPECTED_VAL_ROWS, (val_keys.height, meta["n_val"])
    n_val_matches = val_keys["match_key"].n_unique()
    assert n_val_matches == EXPECTED_VAL_MATCHES, n_val_matches
    feats = SW.cell_features(a.arm, meta["base"])
    assert len(feats) == EXPECTED_FEATS[a.arm], (a.arm, len(feats))
    print(f"arm {a.arm}: {len(feats)} inputs; matrix {X.shape}; fit {meta['n_fit']:,}, early stop {meta['n_es']:,}, "
          f"validation {meta['n_val']:,} rows / {n_val_matches:,} matches; rss {SW.rss_mb()} MB", flush=True)

    data = {"X": X, "y": y, "feats_all": meta["feats_all"], "base": meta["base"], "n_fit": meta["n_fit"], "n_es": meta["n_es"],
            "val_keys": val_keys}
    t_arm = time.time()
    prep = SW.prepare_set(data, a.arm)
    for seed in seeds:
        set_id = a.arm.replace("+", "-")
        cell = {"id": f"{set_id}_d{DEPTH}_e{ETA}_m{MCW}_s{seed}", "set": a.arm, "depth": DEPTH, "eta": ETA, "mcw": MCW, "seed": seed}
        t0 = time.time()
        r = SW.fit_cell_prepared(prep, cell, "cuda", pred_dir, data)
        r.update(cell)
        r.update({"status": "ok", "device": "cuda", "inputs": len(feats), "wall_seconds": round(time.time() - t0, 1),
                  "matrix": MATRIX, "pinned": {"depth": DEPTH, "eta": ETA, "min_child_weight": MCW, "subsample": 0.8,
                                              "tree_method": "hist", "objective": "binary:logistic", "early_stopping_rounds": 100,
                                              "num_boost_round_cap": 3000, "max_bin": 256}})
        with open(os.path.join(res_dir, cell["id"] + ".json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"{cell['id']}: val_logloss {r['val_logloss']:.6f} best {r['best_iteration']} fit {r['fit_seconds']}s "
              f"wall {r['wall_seconds']}s n_val {r['n_validation']:,}", flush=True)
        if a.arm == "base" and seed == 0:
            gap = r["val_logloss"] - PINNED_VAL_LL_GPU_BASE_S0
            print(f"REPRODUCTION base seed 0: {r['val_logloss']:.6f} vs pinned {PINNED_VAL_LL_GPU_BASE_S0} (gap {gap:+.6f}, "
                  f"tolerance {REPRO_TOL})", flush=True)
            if abs(gap) > REPRO_TOL:
                print("REPRODUCTION FAILED: stopping, per protocol", flush=True)
                sys.exit(3)
    del prep
    gc.collect()
    print(f"arm {a.arm} done in {time.time() - t_arm:.0f}s", flush=True)


if __name__ == "__main__":
    main()
