"""A8 check 5: optimism of the raw XGBoost validation log loss (early stopping on validation).

Primary split, seed 0. The model is fitted with the bake-off settings (eval set = validation).
The validation log loss at the chosen iteration (0.43511) is the minimum over 362 candidate
iteration counts on the same validation matches, so it is optimistic. Split-half estimate:
validation matches are split into halves A and B (seeded, independent of the model). For each
half h, k_h is the iteration count minimising validation LL on the other half; the honest LL for
half h is the LL at k_h on half h. The honest estimate is the match-weighted mean over the two
halves, compared with the reported minimum over the full validation set.
Also reports the test log loss at k_h for reference only (not used for any choice).
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_val_optimism.py
"""

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ll_vec(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    xgbs = load("xgb_primary_seed0", ROOT / "scripts" / "model" / "xgb_primary_seed0.py")
    bo = xgbs.bo
    D = xgbs.build_data(dedupe=True)
    S, sets = D["S"], D["sets"]
    tr = S.filter(pl.col("match_key").is_in(sorted(sets["train"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sets["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
    Xtr, ytr = bo.xy(tr)
    Xva, yva = bo.xy(va)
    Xte, yte = bo.xy(te)
    mX = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                           subsample=0.8, tree_method="hist", device="cpu",
                           monotone_constraints=tuple(bo.MONO), early_stopping_rounds=100,
                           eval_metric="logloss", random_state=bo.SEED)
    mX.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bx = int(mX.best_iteration)
    n_trees = mX.get_booster().num_boosted_rounds()
    print(f"best_iteration {bx}; stored trees {n_trees}")

    ks = list(range(1, n_trees + 1))
    dva = xgb.DMatrix(Xva)
    dte = xgb.DMatrix(Xte)
    # per-state losses at each iteration count
    L_va = np.empty((len(ks), len(yva)))
    L_te = np.empty((len(ks), len(yte)))
    for i, k in enumerate(ks):
        L_va[i] = ll_vec(yva, mX.get_booster().predict(dva, iteration_range=(0, k)))
        L_te[i] = ll_vec(yte, mX.get_booster().predict(dte, iteration_range=(0, k)))
    full_curve = L_va.mean(axis=1)
    k_full = ks[int(np.argmin(full_curve))]
    print(f"full-validation minimum: k={k_full} (trees), LL {full_curve.min():.5f} "
          f"(reported 0.43511 at k={bx + 1})")

    # match-level halves of validation, seeded but independent of the model
    matches = sorted(sets["val"])
    random.Random(12345).shuffle(matches)
    half = {m: i % 2 for i, m in enumerate(matches)}
    keys_va = va["match_key"].to_list()
    keys_te = te["match_key"].to_list()
    hv = np.array([half[m] for m in keys_va])
    print(f"validation halves: A {int((hv == 0).sum())} states, B {int((hv == 1).sum())} states")

    honest_num, honest_den = 0.0, 0.0
    test_ref = []
    for h in (0, 1):
        other = 1 - h
        curve_other = L_va[:, hv == other].mean(axis=1)
        i_star = int(np.argmin(curve_other))
        k_star = ks[i_star]
        mask_h = hv == h
        ll_h = float(L_va[i_star, mask_h].mean())
        n_h = int(mask_h.sum())
        honest_num += ll_h * n_h
        honest_den += n_h
        ll_te = float(L_te[i_star].mean())
        test_ref.append((h, k_star, ll_h, ll_te))
        print(f"half {'AB'[h]}: k chosen on half {'AB'[other]} = {k_star}; LL on half {'AB'[h]} at k = {ll_h:.5f}; "
              f"test LL at this k (reference only) = {ll_te:.5f}")
    honest = honest_num / honest_den
    print(f"honest validation LL (split-half, early stopping re-chosen out of sample): {honest:.5f}")
    print(f"reported (in-sample minimum) validation LL: {full_curve.min():.5f}")
    print(f"optimism of the reported validation LL: {honest - full_curve.min():+.5f}")
    print(f"test LL at reported k={bx + 1}: {L_te[bx].mean():.5f}")
    flat = full_curve[max(0, bx - 50): bx + 51]
    print(f"validation LL within +/-50 iterations of the optimum ranges {flat.min():.5f} to {flat.max():.5f}")


if __name__ == "__main__":
    main()
