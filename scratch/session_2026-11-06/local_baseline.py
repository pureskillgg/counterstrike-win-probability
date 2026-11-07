"""Quick local XGBoost baseline on the compact export (laptop, GPU if available). Read-only on the export.

Rows = seconds of 5v5 rounds (every 8th second to keep memory small), label = CT wins.
Model A: alive counts, CT-T alive difference, bomb state, seconds since round live, seconds since plant.
Model B: A plus the per-round loadouts (CT and T, 15 features each) from loadout_round.npy.
Split: match_split.npy (0 train, 1 val, 2 test). Rounds flagged with duplicated player ids are excluded.
Early stopping on validation; reports log loss and accuracy on validation and test.
"""
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb

E = sys.argv[1] if len(sys.argv) > 1 else r"G:\pureskill\csds-cs2\data\export_clean"
STEP = 8
t0 = time.time()

off = np.load(os.path.join(E, "offsets.npy"))
alive = np.load(os.path.join(E, "steps_alive.npy"))
freeze = np.load(os.path.join(E, "round_freeze.npy")).astype(np.int64)
plant = np.load(os.path.join(E, "round_plant.npy")).astype(np.int64)
label = np.load(os.path.join(E, "round_label.npy")).astype(np.float32)
rmatch = np.load(os.path.join(E, "round_match.npy")).astype(np.int64)
loadout = np.load(os.path.join(E, "loadout_round.npy")).astype(np.float32)
flag = np.load(os.path.join(E, "loadout_flag.npy"))
msplit = np.load(os.path.join(E, "match_split.npy"))
names = json.load(open(os.path.join(E, "loadout_names.json")))

N = len(label)
n_per = np.diff(off)
ridx = np.repeat(np.arange(N, dtype=np.int32), n_per)
k = (np.arange(off[-1], dtype=np.int64) - off[ridx]).astype(np.int32)
keep = (k % STEP == 0) & (flag[ridx] == 0)
ridx, k = ridx[keep], k[keep]
ts = freeze[ridx] + 64 * k.astype(np.int64)
pl_ = plant[ridx]
planted = (pl_ >= 0) & (ts >= pl_)
sec_plant = np.where(planted, (ts - pl_) / 64.0, -1.0).astype(np.float32)
ac = alive[keep, 0].astype(np.float32)
at = alive[keep, 1].astype(np.float32)
base = np.stack([ac, at, ac - at, planted.astype(np.float32), k.astype(np.float32), sec_plant], axis=1)
base_names = ["alive_ct", "alive_t", "ct_minus_t", "bomb", "sec_live", "sec_plant"]
X_load = loadout[ridx]
split = msplit[rmatch[ridx]]
y = label[ridx]
print("rows", len(y), "| train", int((split == 0).sum()), "val", int((split == 1).sum()), "test", int((split == 2).sum()),
      f"| secs {time.time() - t0:.0f}", flush=True)


def logloss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(name, X, feat_names):
    tr, va, te = split == 0, split == 1, split == 2
    dtr = xgb.DMatrix(X[tr], label=y[tr], feature_names=feat_names)
    dva = xgb.DMatrix(X[va], label=y[va], feature_names=feat_names)
    dte = xgb.DMatrix(X[te], label=y[te], feature_names=feat_names)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.1, "max_depth": 8,
              "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "device": "cuda"}
    try:
        bst = xgb.train(params, dtr, num_boost_round=2000, evals=[(dva, "val")],
                        early_stopping_rounds=50, verbose_eval=False)
    except xgb.core.XGBoostError:
        params["device"] = "cpu"
        bst = xgb.train(params, dtr, num_boost_round=2000, evals=[(dva, "val")],
                        early_stopping_rounds=50, verbose_eval=False)
    pv, pt = bst.predict(dva), bst.predict(dte)
    res = {"model": name, "device": params["device"], "best_iter": int(bst.best_iteration),
           "val_logloss": round(logloss(pv, y[va]), 4), "val_acc": round(float(((pv > 0.5) == y[va]).mean()), 4),
           "test_logloss": round(logloss(pt, y[te]), 4), "test_acc": round(float(((pt > 0.5) == y[te]).mean()), 4)}
    print(json.dumps(res), f"| secs {time.time() - t0:.0f}", flush=True)
    return res


resA = run("A seconds+alive", base, base_names)
resB = run("B +per-round loadouts", np.concatenate([base, X_load], axis=1), base_names + names)
json.dump({"A": resA, "B": resB, "step": STEP}, open(os.path.join(E, "results_local_baseline.json"), "w"), indent=1)
print("done", f"{time.time() - t0:.0f}s", flush=True)
