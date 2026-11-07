"""A9 audit: fit-to-fit spread of the learning-curve 50% and 100% rows.

The learning curve's confidence intervals resample test matches only. This script measures how much
the test log loss moves when only the XGBoost random_state (row subsampling) or the thread count changes,
on the same nested training subsets as scripts/model/sufficiency/learning_curve.py.

Read-only. Default threads (n_jobs unset, the bake-off setting) for random_state 0..4; plus the
learning-curve setting (n_jobs=4) at seed 0 as a check that the subsets reproduce learning_curve.json
(50%: 0.43800409787819106; 100%: 0.4365863847669546).

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_seed_spread.py
"""

import collections
import importlib.util
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
EX_PATH = ROOT / "scripts" / "eda" / "extract_state_1hz.py"
SPLIT_SEED = 0
FRACS = [0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", EX_PATH)
bk = ex.bk
FEATS, MONO = bk.FEATS, bk.MONO


def xy(df):
    return df.select(FEATS).to_numpy().astype(np.float64), df["label_ct_win"].to_numpy().astype(np.int64)


def fit(tr, va, te, seed, n_jobs):
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, yte = xy(te)
    kw = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
              subsample=0.8, tree_method="hist", device="cpu", monotone_constraints=tuple(MONO),
              early_stopping_rounds=100, eval_metric="logloss", random_state=seed)
    if n_jobs is not None:
        kw["n_jobs"] = n_jobs
    m = xgb.XGBClassifier(**kw)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bx = int(m.best_iteration)
    p = m.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    return bk.metrics(yte, p)["logloss"], bx


def main():
    t0 = time.time()
    kept = ex.bakeoff_kept()
    H = kept["H"]
    rows = []
    for k in kept["rounds_ok"]:
        rr = {**kept["rd"][k], "freeze": kept["live"][k]["tick"], "t_score": kept["live"][k]["t"],
              "ct_score": kept["live"][k]["ct"]}
        rows.extend(bk.build_states(k, rr, kept["deaths"].get(k, []), kept["plants"].get(k, []),
                                    kept["players"].get(k, {}), H[k[0]]["max_rounds"], collections.Counter()))
    S = pl.DataFrame(rows)
    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    order = sorted(sp["train"])
    random.Random(SPLIT_SEED).shuffle(order)
    subsets = {f: order[: int(f * len(order) + 0.5)] for f in FRACS}
    va = S.filter(pl.col("match_key").is_in(sorted(sp["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sp["test"])))

    res = {}
    for f in FRACS:
        tr = S.filter(pl.col("match_key").is_in(sorted(subsets[f])))
        for seed in SEEDS:
            ll, bx = fit(tr, va, te, seed, None)
            res[(f, seed, "default")] = (ll, bx)
            print(f"frac {f} seed {seed} default threads: test LL {ll:.10f} best_iter {bx}", flush=True)
        ll, bx = fit(tr, va, te, 0, 4)
        res[(f, 0, "nj4")] = (ll, bx)
        print(f"frac {f} seed 0 n_jobs=4 (learning_curve setting): test LL {ll:.10f} best_iter {bx}", flush=True)

    print("\n== summary ==", flush=True)
    for f in FRACS:
        lls = np.array([res[(f, s, "default")][0] for s in SEEDS])
        bxs = [res[(f, s, "default")][1] for s in SEEDS]
        print(f"frac {f}: default threads over seeds {SEEDS}: mean LL {lls.mean():.6f}, sd {lls.std(ddof=1):.6f}, "
              f"range {lls.max() - lls.min():.6f}; best_iter {bxs}", flush=True)
    d = np.array([res[(1.0, s, "default")][0] - res[(0.5, s, "default")][0] for s in SEEDS])
    print(f"change 50%->100% per seed (default threads): {np.round(d, 6).tolist()}; mean {d.mean():+.6f}, "
          f"sd {d.std(ddof=1):.6f}; seeds where 100% is better: {int((d < 0).sum())} of {len(d)}", flush=True)
    print(f"learning_curve.json: 50% 0.43800409787819106 (n_jobs=4), 100% 0.4365863847669546 (n_jobs=4)", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
