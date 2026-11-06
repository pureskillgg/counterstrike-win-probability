"""Drop-group ablations of the 83-input model, on the same matrix, split and fit path as the sprint's full cell.

Why: the sprint's ablations only add groups to the 59-input base. This script removes inputs from the full model, and it fits
two sets that answer the open alive-count question directly:

  full                 all 83 inputs (reference, refit here so every comparison uses one environment)
  minus_alive          full without alive_ct, alive_t, ct_alive_minus_t
  round_state_only     the six ROUND_STATE inputs and nothing else
  loadout_only         the 46 st_* loadout inputs and nothing else
  minus_round_state    full without the six ROUND_STATE inputs
  minus_blind          full without the four BLIND inputs
  minus_view           full without the four VIEW inputs
  minus_armor_sums     full without st_armor_sum_ct and st_armor_sum_t
  minus_bomb_geo       full without c_min_bomb_dist, c_n_bomb_500, t_min_bomb_dist, defuse_slack

Fit path, copied from experiments/sprint/sprint_worker.py (fit_cell_prepared and prepare_set): binary:logistic, logloss,
eta 0.03, depth 4, min_child_weight 20, subsample 0.8, hist, max_bin 256, early stopping 100 on the early-stop rows,
3,000-round cap, monotone constraints alive_ct +1, alive_t -1, ct_alive_minus_t +1, seeds 0 to 2. Validation log loss is
computed on the validation rows with clipping at 1e-15.

Pipelining: while the GPU trains one feature set, a CPU thread prepares the next set's quantised matrices, as the sprint
worker does. The results do not depend on this overlap.

Run in WSL on the GPU:
  python3 scripts/explain/xgb83_drop_ablation.py --matrix /mnt/c/Users/13144/sprint/matrix_trainctr \
      --out /mnt/c/Users/13144/sprint/explain83_ablation --device cuda
A smoke test on small slices (no conclusions): add --smoke. Results that already exist are skipped, so a rerun resumes.
"""
import argparse
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xgboost as xgb

ALIVE = ["alive_ct", "alive_t", "ct_alive_minus_t"]
ROUND_STATE = ["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"]
BLIND = ["c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]
VIEW = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
ARMOR_SUMS = ["st_armor_sum_ct", "st_armor_sum_t"]
BOMB_GEO = ["c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist", "defuse_slack"]
MONO = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1}  # same as wp_common.MONO_ALIVE
SPRINT_FULL = "/mnt/g/pureskill/csds-cs2/data/sprint/run/gpu_fast/results/full_d4_e0.03_m20_s{seed}.json"
SEEDS = [0, 1, 2]


def build_sets(feats):
    full = list(feats)
    loadout = [f for f in feats if f.startswith("st_")]
    sets = {
        "full": full,
        "minus_alive": [f for f in full if f not in ALIVE],
        "round_state_only": [f for f in full if f in ROUND_STATE],
        "loadout_only": loadout,
        "minus_round_state": [f for f in full if f not in ROUND_STATE],
        "minus_blind": [f for f in full if f not in BLIND],
        "minus_view": [f for f in full if f not in VIEW],
        "minus_armor_sums": [f for f in full if f not in ARMOR_SUMS],
        "minus_bomb_geo": [f for f in full if f not in BOMB_GEO],
    }
    assert len(sets["loadout_only"]) == 46, f"expected 46 loadout inputs, got {len(sets['loadout_only'])}"
    assert len(sets["round_state_only"]) == 6 and len(sets["minus_bomb_geo"]) == 79
    return sets


def logloss(p, y):
    p = np.clip(p.astype(np.float64), 1e-15, 1)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--smoke", action="store_true", help="small slices of each split, to test the code path only")
    ap.add_argument("--only", nargs="*", help="run only these sets")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.matrix, "meta.json")))
    feats = meta["feats_all"]
    # memory-mapped from WSL's own disk (copy the matrix there first; mapping over /mnt/c pages in 4 KB at a time is far too
    # slow). Loading all 5 GB into RAM and keeping two feature sets in memory at once was killed by the OOM killer.
    X = np.load(os.path.join(a.matrix, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(a.matrix, "y.npy"), mmap_mode="r")
    assert X.shape == (meta["rows"], len(feats)), "matrix shape does not match meta.json"
    a1, a2 = meta["n_fit"], meta["n_fit"] + meta["n_es"]
    spans = {"fit": (0, a1), "es": (a1, a2), "val": (a2, meta["rows"])}
    if a.smoke:
        spans = {"fit": (0, 200_000), "es": (a1, a1 + 50_000), "val": (a2, a2 + 50_000)}
    res_dir, pred_dir = os.path.join(a.out, "results"), os.path.join(a.out, "preds")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    sets = build_sets(feats)
    json.dump(sets, open(os.path.join(a.out, "sets.json"), "w"), indent=1)
    names = a.only or list(sets)
    todo = [(n, [s for s in SEEDS if not os.path.exists(os.path.join(res_dir, f"{n}_s{s}.json"))]) for n in names]
    todo = [(n, seeds) for n, seeds in todo if seeds]
    print(f"xgboost {xgb.__version__}; device {a.device}; matrix {X.shape}; sets to run {[n for n, _ in todo]}", flush=True)

    def prepare(name):
        cols = [feats.index(f) for f in sets[name]]
        t0 = time.time()
        f0, f1 = spans["fit"]
        e0, e1 = spans["es"]
        v0, v1 = spans["val"]
        dtrain = xgb.QuantileDMatrix(np.ascontiguousarray(X[f0:f1][:, cols]), np.asarray(y[f0:f1]), max_bin=256)
        dval = xgb.QuantileDMatrix(np.ascontiguousarray(X[e0:e1][:, cols]), np.asarray(y[e0:e1]), ref=dtrain, max_bin=256)
        yva = np.asarray(y[v0:v1])
        dva = xgb.QuantileDMatrix(np.ascontiguousarray(X[v0:v1][:, cols]), yva, ref=dtrain, max_bin=256)
        quant_s = round(time.time() - t0, 1)
        print(f"{name}: {len(cols)} inputs quantised in {quant_s}s", flush=True)
        return {"name": name, "dtrain": dtrain, "dval": dval, "dva": dva, "yva": yva, "quant_s": quant_s,
                "mono": tuple(MONO.get(f, 0) for f in sets[name])}

    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(prepare, todo[0][0]) if todo else None
    for i, (name, seeds) in enumerate(todo):
        prep = pending.result()
        pending = pool.submit(prepare, todo[i + 1][0]) if i + 1 < len(todo) else None
        for seed in seeds:
            params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.03, "max_depth": 4,
                      "min_child_weight": 20, "subsample": 0.8, "tree_method": "hist", "device": a.device,
                      "seed": seed, "monotone_constraints": prep["mono"]}
            t1 = time.time()
            bst = xgb.train(params, prep["dtrain"], num_boost_round=3000, evals=[(prep["dval"], "validation_0")],
                            early_stopping_rounds=100, verbose_eval=False)
            fit_s = round(time.time() - t1, 1)
            best = int(bst.best_iteration)
            p = bst.predict(prep["dva"], iteration_range=(0, best + 1))
            ll = logloss(p, prep["yva"])
            np.save(os.path.join(pred_dir, f"{name}_s{seed}.npy"), p.astype(np.float32))
            out = {"set": name, "seed": seed, "n_inputs": len(sets[name]), "best_iteration": best, "val_logloss": ll,
                   "fit_seconds": fit_s, "quantise_seconds": prep["quant_s"], "n_validation": int(len(prep["yva"])),
                   "xgboost": xgb.__version__, "device": a.device, "smoke": a.smoke}
            if name == "full" and not a.smoke:
                ref = SPRINT_FULL.format(seed=seed)
                if os.path.exists(ref):
                    out["sprint_val_logloss_same_seed"] = json.load(open(ref))["val_logloss"]
            json.dump(out, open(os.path.join(res_dir, f"{name}_s{seed}.json"), "w"), indent=1)
            print(f"{name} seed {seed}: val_logloss {ll:.6f} best {best} fit {fit_s}s", flush=True)
        del prep
        gc.collect()


if __name__ == "__main__":
    main()
