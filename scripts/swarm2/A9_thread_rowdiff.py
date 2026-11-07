"""A9 audit: how far does XGBoost thread count (n_jobs) move the 100%-row model?

Read-only. Same kept set, split and 100% train set as scripts/model/sufficiency/learning_curve.py.
For each n_jobs setting (None = bake-off default), fits device="cpu" and records:
  best iteration, test log loss (10 dp), max and 99th percentile |p - p_default| on the 47,716 test
  states, number of test states with |dp| > 0.01, and tree structure agreement with the default fit
  (split_indices equal, split_conditions equal, leaf weights equal, per tree).
Also compares the default fit with the saved bake-off model models/xgb_primary_seed0.json.

Usage (project root): python.exe -B -I -X utf8 scripts\\swarm2\\A9_thread_rowdiff.py
"""

import collections
import importlib.util
import json
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
SAVED_MODEL = ROOT / "models" / "xgb_primary_seed0.json"
SEED = 0
NJ_SETTINGS = [None, 1, 2, 4, 8, 16]


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


def trees_fields(booster):
    raw = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    out = []
    for t in raw["learner"]["gradient_booster"]["model"]["trees"]:
        out.append({k: json.dumps(t[k]) for k in ("split_indices", "split_conditions",
                                                  "left_children", "base_weights")})
    return out


def agree(ta, tb):
    n = min(len(ta), len(tb))
    c = collections.Counter()
    for a, b in zip(ta[:n], tb[:n]):
        for k in a:
            c[k] += int(a[k] == b[k])
    return {k: f"{v}/{n}" for k, v in c.items()}


def main():
    t0 = time.time()
    kept = ex.bakeoff_kept()
    H = kept["H"]
    state_rows = []
    for k in kept["rounds_ok"]:
        rr = {**kept["rd"][k], "freeze": kept["live"][k]["tick"], "t_score": kept["live"][k]["t"],
              "ct_score": kept["live"][k]["ct"]}
        out = bk.build_states(k, rr, kept["deaths"].get(k, []), kept["plants"].get(k, []),
                              kept["players"].get(k, {}), H[k[0]]["max_rounds"], collections.Counter())
        state_rows.extend(out)
    S = pl.DataFrame(state_rows)
    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    tr = S.filter(pl.col("match_key").is_in(sorted(sp["train"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sp["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sp["test"])))
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, yte = xy(te)
    print(f"train {len(ytr)} states, val {len(yva)}, test {len(yte)}; data built in {time.time() - t0:.1f}s",
          flush=True)

    saved = xgb.XGBClassifier()
    saved.load_model(str(SAVED_MODEL))
    p_saved = saved.predict_proba(Xte, iteration_range=(0, 262))[:, 1]
    trees_saved = trees_fields(saved.get_booster())

    results = {}
    for nj in NJ_SETTINGS:
        kw = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                  subsample=0.8, tree_method="hist", device="cpu", monotone_constraints=tuple(MONO),
                  early_stopping_rounds=100, eval_metric="logloss", random_state=SEED)
        if nj is not None:
            kw["n_jobs"] = nj
        t1 = time.time()
        m = xgb.XGBClassifier(**kw)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        bx = int(m.best_iteration)
        p = m.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
        results[nj] = {"p": p, "bx": bx, "trees": trees_fields(m.get_booster()),
                       "fit_s": round(time.time() - t1, 1)}
        print(f"n_jobs={nj}: best_iter {bx}, fit {results[nj]['fit_s']}s", flush=True)

    base = results[None]
    print("\n== per setting vs default (n_jobs unset) ==", flush=True)
    for nj in NJ_SETTINGS:
        r = results[nj]
        d = np.abs(r["p"] - base["p"])
        ll = bk.metrics(yte, r["p"])["logloss"]
        print(f"n_jobs={str(nj):>4}: test LL {ll:.10f}; max|dp| {d.max():.4e}; p99|dp| {np.percentile(d, 99):.4e}; "
              f"states |dp|>0.01: {int((d > 0.01).sum())}; "
              f"tree fields equal to default (of min trees): {agree(base['trees'], r['trees'])}", flush=True)

    print("\n== saved bake-off model vs default fit ==", flush=True)
    d = np.abs(p_saved - base["p"])
    print(f"saved model test LL {bk.metrics(yte, p_saved)['logloss']:.10f}; default fit vs saved: "
          f"max|dp| {d.max():.4e}; trees: {agree(trees_saved, base['trees'])}", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
