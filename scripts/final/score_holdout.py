"""Pin 1b7c86fd. Step 1: reproduction gate on validation for the candidate (XGBoost full set, depth 4, eta 0.03, mcw 20, seeds 0 to 2).
Step 2, only if all three seeds pass: predict the Oct 26-31 holdout feature matrix. No holdout label is read here.
Training call and QuantileDMatrix path are those of experiments/sprint/sprint_worker.py (prepare_set, fit_cell_prepared)."""
import json
import os
import sys

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = "/mnt/g/pureskill/csds-cs2"
SPRINT = "/mnt/c/Users/<user>/Documents/pureskill/counterstrike-win-probability/experiments/sprint"
os.environ["CS2_ROOT"] = ROOT
sys.path.insert(0, f"{ROOT}/scripts/model")
sys.path.insert(0, SPRINT)
import sprint_worker as SW  # noqa: E402

MAT = "/home/<user>/sprint/matrix_trainctr"
HOLD = "/mnt/c/Users/<user>/sprint/holdout_oct26_31"
SAVED = f"{ROOT}/data/sprint/run/gpu_fast/preds"
PINNED_LL = {0: 0.450228, 1: 0.450305, 2: 0.450223}  # 88d9fbb0 / 179e83e4 comparison values, 6 dp

meta = json.load(open(f"{MAT}/meta.json"))
feats = meta["feats_all"]
assert SW.cell_features("full", meta["base"]) == feats, "full-set feature order differs from the stored matrix"
X = np.load(f"{MAT}/X.npy", mmap_mode="r")
y = np.load(f"{MAT}/y.npy", mmap_mode="r")
data = {"X": X, "y": y, "n_fit": meta["n_fit"], "n_es": meta["n_es"], "feats_all": feats, "base": meta["base"],
        "val_keys": pl.read_parquet(f"{MAT}/val_keys.parquet")}
prep = SW.prepare_set(data, "full")
print("quantised full set", flush=True)

gate = {}
boosters = {}
for s in (0, 1, 2):
    c = {"id": f"full_d4_e0.03_m20_s{s}", "set": "full", "depth": 4, "eta": 0.03, "mcw": 20, "seed": s}
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": c["eta"], "max_depth": c["depth"],
              "min_child_weight": c["mcw"], "subsample": 0.8, "tree_method": "hist", "device": "cuda", "seed": c["seed"],
              "monotone_constraints": tuple(SW.W.mono_for(prep["feats"]))}
    bst = xgb.train(params, prep["dtrain"], num_boost_round=3000, evals=[(prep["dval"], "validation_0")],
                    early_stopping_rounds=100, verbose_eval=False)
    best = int(bst.best_iteration)
    p = bst.predict(prep["dva"], iteration_range=(0, best + 1)).astype(np.float64)
    yv = prep["yv"].astype(np.float64)
    ll = float(-(yv * np.log(np.clip(p, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1))).mean())
    saved = pl.read_parquet(f"{SAVED}/{c['id']}.parquet")
    assert saved.height == p.size, "saved validation file has a different row count"
    ps = saved["p"].to_numpy().astype(np.float64)
    ll_saved = float(-(yv * np.log(np.clip(ps, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - ps, 1e-15, 1))).mean())
    maxdiff = float(np.abs(p - ps).max())
    ok = bool(maxdiff < 1e-5 and abs(ll - ll_saved) < 1e-6)
    gate[s] = {"best_iteration": best, "val_ll_refit": ll, "val_ll_saved": ll_saved, "pinned_6dp": PINNED_LL[s],
               "max_abs_diff_p": maxdiff, "pass": ok}
    print(f"seed {s}: best {best}, refit LL {ll:.8f}, saved LL {ll_saved:.8f}, pinned {PINNED_LL[s]}, max|dp| {maxdiff:.3e}, pass {ok}",
          flush=True)
    boosters[s] = bst

json.dump({"gate": gate, "pin": "1b7c86fd", "fit_path": SW.FIT_PATH}, open(f"{HOLD}/score_gate.json", "w"), indent=1)
if not all(g["pass"] for g in gate.values()):
    print("GATE FAILED: no holdout prediction made. Stop and amend pin 1b7c86fd first.", flush=True)
    sys.exit(3)

print("gate passed for all three seeds; predicting the holdout feature matrix (no labels read)", flush=True)
Xh = np.load(f"{HOLD}/X.npy")
assert Xh.shape == (4320801, 83), f"holdout matrix shape {Xh.shape}"
dh = xgb.QuantileDMatrix(Xh, label=np.zeros(Xh.shape[0], dtype=np.float32), ref=prep["dtrain"], max_bin=256)
for s, bst in boosters.items():
    best = gate[s]["best_iteration"]
    ph = bst.predict(dh, iteration_range=(0, best + 1)).astype(np.float64)
    np.save(f"{HOLD}/p_seed{s}.npy", ph)
    print(f"seed {s}: holdout predictions saved, n {ph.size}, mean p {ph.mean():.4f}", flush=True)
print("done", flush=True)
