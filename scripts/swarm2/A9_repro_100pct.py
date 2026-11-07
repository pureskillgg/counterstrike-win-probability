"""A9 audit: reproduce the learning-curve 100% row and compare it with the saved bake-off model.

Read-only with respect to the project. Writes nothing except what the caller redirects to a log.
Reads: tomes/2026-10-01_to_08 (via bakeoff/feature_validation), models/xgb_primary_seed0.json (read only).

Variants (one process each; the fit is the only thing that differs):
  lc_exact     learning_curve.py call path: device="cuda" request (falls back to CPU), n_jobs=4,
               OMP_NUM_THREADS=4, POLARS_MAX_THREADS=4.
  cpu_default  bakeoff.py / xgb_primary_seed0.py call path: device="cpu", no n_jobs, no thread env.
  cpu_nj4      device="cpu", n_jobs=4, thread env 4. Isolates the n_jobs/thread setting from the
               device request.

Every variant rebuilds the kept set and states with the same code path as learning_curve.py, uses
the bake-off primary split (seed 0), and trains on all 1095 train-pool matches (the 100% row).
It then scores the test set (304 matches) and compares with the saved model
models/xgb_primary_seed0.json (best_iteration 261, metrics_test.logloss 0.4365987668076211).

Usage (project root):
  python.exe -B -I -X utf8 scripts\\swarm2\\A9_repro_100pct.py <variant>
"""

import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "cpu_default"
assert VARIANT in ("lc_exact", "cpu_default", "cpu_nj4"), VARIANT
if VARIANT in ("lc_exact", "cpu_nj4"):
    # same environment lines as learning_curve.py, set before numpy/polars/xgboost import
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["POLARS_MAX_THREADS"] = "4"

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
EX_PATH = ROOT / "scripts" / "eda" / "extract_state_1hz.py"
SAVED_MODEL = ROOT / "models" / "xgb_primary_seed0.json"
SAVED_META = ROOT / "models" / "xgb_primary_seed0_meta.json"
SEED = 0
EPS = 1e-15


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", EX_PATH)  # imports bakeoff.py read only; its main() is not run
bk = ex.bk
FEATS, MONO = bk.FEATS, bk.MONO


def xy(df):
    X = df.select(FEATS).to_numpy().astype(np.float64)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def fit_variant(tr, va, te):
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, _ = xy(te)
    if VARIANT == "lc_exact":
        device, n_jobs = "cuda", 4
    elif VARIANT == "cpu_nj4":
        device, n_jobs = "cpu", 4
    else:
        device, n_jobs = "cpu", None
    kw = dict(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
              subsample=0.8, tree_method="hist", device=device,
              monotone_constraints=tuple(MONO), early_stopping_rounds=100,
              eval_metric="logloss", random_state=SEED)
    if n_jobs is not None:
        kw["n_jobs"] = n_jobs
    m = xgb.XGBClassifier(**kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    fell_back = [str(w.message)[:160] for w in caught if "Device is changed" in str(w.message)]
    bx = int(m.best_iteration)
    p_te = m.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    return m, p_te, bx, device, n_jobs, fell_back


def trees_of(booster):
    """Structural content of every tree, as lists, for an exact comparison."""
    raw = json.loads(booster.save_raw(raw_format="json").decode("utf-8"))
    trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    keys = ("split_indices", "split_conditions", "left_children", "right_children", "base_weights")
    return [tuple(json.dumps(t[k]) for k in keys) for t in trees]


def main():
    t0 = time.time()
    print(f"== A9 reproduce 100% row, variant {VARIANT} ==", flush=True)
    print(f"xgboost {xgb.__version__}, polars {pl.__version__}, numpy {np.__version__}", flush=True)
    print(f"env OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"POLARS_MAX_THREADS={os.environ.get('POLARS_MAX_THREADS')}", flush=True)

    kept = ex.bakeoff_kept()
    H, got = kept["H"], kept["got"]
    print(f"kept set: {got}", flush=True)

    # Same state build as learning_curve.py main() (steps 138-146).
    import collections
    state_rows = []
    for k in kept["rounds_ok"]:
        rr = {**kept["rd"][k], "freeze": kept["live"][k]["tick"], "t_score": kept["live"][k]["t"],
              "ct_score": kept["live"][k]["ct"]}
        out = bk.build_states(k, rr, kept["deaths"].get(k, []), kept["plants"].get(k, []),
                              kept["players"].get(k, {}), H[k[0]]["max_rounds"], collections.Counter())
        assert out is not None
        state_rows.extend(out)
    S = pl.DataFrame(state_rows)
    assert S.height == got["states"] and S["match_key"].n_unique() == got["kept_matches"]

    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    test_m, val_m, train_m = sp["test"], sp["val"], sp["train"]
    print(f"splits: train {len(train_m)}, val {len(val_m)}, test {len(test_m)}", flush=True)

    tr = S.filter(pl.col("match_key").is_in(sorted(train_m)))  # 100% row, S order (as in both scripts)
    va = S.filter(pl.col("match_key").is_in(sorted(val_m)))
    te = S.filter(pl.col("match_key").is_in(sorted(test_m)))
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    print(f"train states {tr.height}, val {va.height}, test {te.height}", flush=True)
    print(f"first 3 train rows (match_key, round): {tr.select(['match_key', 'round']).head(3).rows()}",
          flush=True)

    # Saved bake-off model: predictions on this test set.
    saved = xgb.XGBClassifier()
    saved.load_model(str(SAVED_MODEL))
    saved_meta = json.loads(SAVED_META.read_text(encoding="utf-8"))
    saved_bx = saved_meta["best_iteration"]
    Xte, _ = xy(te)
    p_saved = saved.predict_proba(Xte, iteration_range=(0, saved_bx + 1))[:, 1]
    met_saved = bk.metrics(y_te, p_saved)
    print(f"SAVED model (bake-off): best_iter {saved_bx}, test LL {met_saved['logloss']:.10f}, "
          f"Brier {met_saved['brier']:.10f} (meta file LL {saved_meta['metrics_test']['logloss']:.10f})",
          flush=True)

    print("== fit ==", flush=True)
    m, p_te, bx, device, n_jobs, fell_back = fit_variant(tr, va, te)
    met = bk.metrics(y_te, p_te)
    print(f"variant {VARIANT}: device requested {device}, n_jobs {n_jobs}, best_iter {bx}", flush=True)
    print(f"  device-fallback warnings: {len(fell_back)} {fell_back[:1]}", flush=True)
    print(f"  test LL {met['logloss']:.10f}  Brier {met['brier']:.10f}  AUC {met['auc']:.6f}", flush=True)
    print(f"  test LL 5dp {met['logloss']:.5f}  (bake-off log: 0.43660; learning_curve.json 100%: "
          f"0.4365863847669546)", flush=True)
    print(f"  vs SAVED model: max |p diff| on test = {np.max(np.abs(p_te - p_saved)):.3e}; "
          f"LL diff = {met['logloss'] - met_saved['logloss']:+.10f}", flush=True)

    tr_saved = trees_of(saved.get_booster())
    tr_this = trees_of(m.get_booster())
    same = [a == b for a, b in zip(tr_saved, tr_this)]
    print(f"  trees: saved {len(tr_saved)}, this run {len(tr_this)}; identical structure "
          f"(split index/condition, children, leaf weights) in {sum(same)} of "
          f"{min(len(tr_saved), len(tr_this))}; first differing tree: "
          f"{next((i for i, s in enumerate(same) if not s), None)}", flush=True)

    # Did the thread setting alone change the trees? (answered by the three-variant comparison in the report)
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
