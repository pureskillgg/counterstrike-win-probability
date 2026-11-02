"""CHECK A: learning curve on the 2026 kept set (seed 0).

Question: does more training data still lower test log loss?

Splits (bake-off primary split, seed 0, via bakeoff.make_splits)
  test        20 percent of kept matches, fixed.
  validation  10 percent of the remaining 80 percent, fixed. Used only for XGBoost early stopping
              (100 rounds). Never trained on.
  train pool  the other 90 percent of the 80 percent.
Fractions 10, 25, 50, 100 percent of the train pool, nested: a seeded shuffle of the sorted pool,
first n_f matches. Every fraction uses the same validation and test matches, so only the number of
training matches changes.

Reuses (read only)
  scripts/eda/extract_state_1hz.py  bakeoff_kept(): the kept set, checked against logs/bakeoff.log.
  scripts/model/bakeoff.py          make_splits, build_states, metrics, FEATS, MONO.

Run from the project root (-B: no bytecode written, so only the output folders change):
  pixi run python -B -I -X utf8 scripts\\model\\sufficiency\\learning_curve.py
"""

import collections
import importlib.util
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

# The machine is shared with other jobs using many threads; cap this run so OpenMP does not thrash.
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["POLARS_MAX_THREADS"] = "4"

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_DIR = ROOT / "notes" / "model" / "sufficiency"
OUT_JSON = OUT_DIR / "learning_curve.json"
SEED = 0
FRACS = [0.10, 0.25, 0.50, 1.00]
N_BOOT = 2000
# logs/bakeoff.log, primary split, XGBoost raw (CPU run)
BAKEOFF_XGB_TEST = {"logloss": 0.43660, "brier": 0.14637}
EPS = 1e-15


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = load_module("extract_state_1hz", ROOT / "scripts" / "eda" / "extract_state_1hz.py")
bk = ex.bk
FEATS, MONO = bk.FEATS, bk.MONO


def counts(df):
    return {"matches": df["match_key"].n_unique(),
            "rounds": df.select(["match_key", "round"]).unique().height,
            "states": df.height,
            "ct_share": round(float(df["label_ct_win"].mean()), 6)}


def xy(df):
    X = df.select(FEATS).to_numpy().astype(np.float64)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def fit_xgb(tr, va, te, device):
    """Returns (test probabilities, best_iteration, fell_back_to_cpu). XGBoost 3.x does not raise
    when no GPU is found; it warns and runs on CPU, so the warning is read to record the device."""
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    Xte, _ = xy(te)
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                          subsample=0.8, tree_method="hist", device=device,
                          monotone_constraints=tuple(MONO), early_stopping_rounds=100,
                          eval_metric="logloss", random_state=SEED, n_jobs=4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    fell_back = any("Device is changed" in str(w.message) for w in caught)
    bx = int(m.best_iteration)
    p = m.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    return p, bx, fell_back


def fit_try_cuda(tr, va, te):
    try:
        p, bx, fell_back = fit_xgb(tr, va, te, "cuda")
    except Exception as exc:
        print(f"  cuda raised ({type(exc).__name__}: {str(exc)[:160]}); using cpu", flush=True)
        p, bx, _ = fit_xgb(tr, va, te, "cpu")
        return p, bx, "cpu (cuda raised)"
    if fell_back:
        return p, bx, "cpu (cuda requested; XGBoost found no GPU and used CPU)"
    return p, bx, "cuda"


def ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_delta(te, p_from, p_to, y):
    """Log loss (to) minus log loss (from), state-weighted, cluster bootstrap over test matches."""
    d = ll_vec(p_to, y) - ll_vec(p_from, y)
    uniq = sorted(te["match_key"].unique().to_list())
    pos = {m: i for i, m in enumerate(uniq)}
    g = np.array([pos[m] for m in te["match_key"].to_list()])
    M = len(uniq)
    sum_d = np.bincount(g, weights=d, minlength=M)
    cnt = np.bincount(g, minlength=M).astype(np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, M, size=(N_BOOT, M))
    boot = sum_d[idx].sum(axis=1) / cnt[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "share_resamples_to_better": float((boot < 0).mean())}


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== CHECK A: learning curve (2026 kept set, seed 0) ==")
    kept = ex.bakeoff_kept()  # prints counts, asserts they reproduce logs/bakeoff.log
    H, got = kept["H"], kept["got"]

    state_rows = []
    for k in kept["rounds_ok"]:
        rr = {**kept["rd"][k], "freeze": kept["live"][k]["tick"], "t_score": kept["live"][k]["t"],
              "ct_score": kept["live"][k]["ct"]}
        out = bk.build_states(k, rr, kept["deaths"].get(k, []), kept["plants"].get(k, []),
                              kept["players"].get(k, {}), H[k[0]]["max_rounds"], collections.Counter())
        assert out is not None, "kept round with unresolved victim side"
        state_rows.extend(out)
    S = pl.DataFrame(state_rows)
    assert S.height == got["states"] and S["match_key"].n_unique() == got["kept_matches"]
    print(f"kept set rebuilt: {S.height} states, {S['match_key'].n_unique()} matches, "
          f"{S.select(['match_key', 'round']).unique().height} rounds")

    sp = bk.make_splits(kept["kept_final"], H)[bk.PRIMARY]
    test_m, val_m, train_m = sp["test"], sp["val"], sp["train"]
    assert not (test_m & val_m) and not (test_m & train_m) and not (val_m & train_m)
    assert test_m | val_m | train_m == kept["kept_final"]

    order = sorted(train_m)
    random.Random(SEED).shuffle(order)
    subsets = {f: order[: int(f * len(order) + 0.5)] for f in FRACS}
    for a, b in zip(FRACS, FRACS[1:]):
        assert set(subsets[a]) <= set(subsets[b]), "fractions are not nested"

    va = S.filter(pl.col("match_key").is_in(sorted(val_m)))
    te = S.filter(pl.col("match_key").is_in(sorted(test_m)))
    y_te = te["label_ct_win"].to_numpy().astype(np.int64)
    split_info = {
        "test": counts(te),
        "validation": counts(va),
        "train_pool": counts(S.filter(pl.col("match_key").is_in(sorted(train_m)))),
    }
    print(f"splits (matches): test {len(test_m)}, validation {len(val_m)}, train pool {len(train_m)}; "
          f"fractions of train pool: {{ {', '.join(f'{f}: {len(subsets[f])}' for f in FRACS)} }}")

    print("== fits ==")
    fr_out, preds = [], {}
    for f in FRACS:
        tr = S.filter(pl.col("match_key").is_in(sorted(subsets[f])))
        t1 = time.time()
        p, bx, dev = fit_try_cuda(tr, va, te)
        met = bk.metrics(y_te, p)
        preds[f] = p
        fr_out.append({"fraction": f, "train": counts(tr), "best_iteration": bx, "device": dev,
                       "fit_s": round(time.time() - t1, 1),
                       "test": {k: met[k] for k in ("n", "logloss", "brier", "auc", "ece", "acc")}})
        print(f"  {int(f * 100):>3}%: train matches {len(subsets[f])}, states {tr.height}, "
              f"best_iter {bx}, device {dev}, test LL {met['logloss']:.5f}, Brier {met['brier']:.5f}, "
              f"AUC {met['auc']:.4f}, ECE {met['ece']:.4f}")

    ll = {r["fraction"]: r["test"]["logloss"] for r in fr_out}
    deltas = []
    for a, b in zip(FRACS, FRACS[1:]):
        r = paired_delta(te, preds[a], preds[b], y_te)
        assert abs(r["delta"] - (ll[b] - ll[a])) < 1e-6, "bootstrap point estimate disagrees"
        deltas.append({"from": a, "to": b, "test_logloss_change": r["delta"],
                       "ci95": r["ci95"], "share_resamples_to_better": r["share_resamples_to_better"]})

    # cross-check: 100 percent row on CPU, same settings as logs/bakeoff.log (CPU run there)
    if fr_out[-1]["device"].startswith("cpu"):
        cpu_p, cpu_bx = preds[1.0], fr_out[-1]["best_iteration"]
    else:
        tr100 = S.filter(pl.col("match_key").is_in(sorted(subsets[1.0])))
        cpu_p, cpu_bx, _ = fit_xgb(tr100, va, te, "cpu")
    cpu_met = bk.metrics(y_te, cpu_p)
    cross = {"bakeoff_log": BAKEOFF_XGB_TEST,
             "here_cpu_100pct": {"logloss": cpu_met["logloss"], "brier": cpu_met["brier"],
                                 "best_iteration": cpu_bx},
             "cpu_matches_bakeoff_log_5dp": round(cpu_met["logloss"], 5) == BAKEOFF_XGB_TEST["logloss"]
             and round(cpu_met["brier"], 5) == BAKEOFF_XGB_TEST["brier"]}
    print(f"cross-check 100% on cpu: LL {cpu_met['logloss']:.5f} (bakeoff log "
          f"{BAKEOFF_XGB_TEST['logloss']:.5f}), Brier {cpu_met['brier']:.5f} (log "
          f"{BAKEOFF_XGB_TEST['brier']:.5f}), best_iter {cpu_bx}; "
          f"matches log: {cross['cpu_matches_bakeoff_log_5dp']}")

    d50 = next(d for d in deltas if d["from"] == 0.5 and d["to"] == 1.0)
    improves = d50["test_logloss_change"] < 0
    ci_excludes_zero = d50["ci95"][1] < 0
    print(f"50% -> 100%: test LL change {d50['test_logloss_change']:+.5f}, "
          f"95% CI [{d50['ci95'][0]:+.5f}, {d50['ci95'][1]:+.5f}]; improves: {improves}; "
          f"CI below zero: {ci_excludes_zero}")

    out = {
        "label": "VERIFIED",
        "check": "A learning curve",
        "seed": SEED,
        "kept_set_check": got,
        "splits": split_info,
        "fractions": fr_out,
        "test_logloss_changes": deltas,
        "crosscheck_100pct_vs_bakeoff": cross,
        "log_loss_50_to_100_improves": improves,
        "ci95_50_to_100_below_zero": ci_excludes_zero,
        "settings": {"xgboost": "hist, depth 4, eta 0.05, min_child_weight 20, subsample 0.8, "
                                "early stopping 100 on validation, monotone MONO",
                     "features": FEATS, "test_frac": 0.20, "validation_frac_of_80pct": 0.10,
                     "bootstrap": f"cluster over test matches, {N_BOOT} resamples, seed {SEED}"},
        "notes": [
            "Validation is 10 percent of the 80 percent pool, so the 100 percent row trains on "
            f"{len(train_m)} of {len(kept['kept_final'])} kept matches (72 percent). A literal 100 percent "
            "of the 80 percent would leave no early-stopping slice.",
            "Fractions are of matches. Metrics are on states (bakeoff metrics()).",
            "Test set and validation set are the bake-off primary split (same seed, same code).",
        ],
        "wall_s": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {OUT_JSON}")

    print("\n| train fraction | train matches | train rounds | train states | XGB best iter | test LL | test Brier | test AUC | test ECE | device |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in fr_out:
        t = r["test"]
        print(f"| {int(r['fraction'] * 100)}% | {r['train']['matches']} | {r['train']['rounds']} | "
              f"{r['train']['states']} | {r['best_iteration']} | {t['logloss']:.5f} | {t['brier']:.5f} | "
              f"{t['auc']:.4f} | {t['ece']:.4f} | {r['device']} |")
    print("\n| change in test LL | delta | 95% CI | share of resamples better |")
    print("|---|---:|---|---:|")
    for d in deltas:
        print(f"| {int(d['from'] * 100)}% -> {int(d['to'] * 100)}% | {d['test_logloss_change']:+.5f} | "
              f"[{d['ci95'][0]:+.5f}, {d['ci95'][1]:+.5f}] | {d['share_resamples_to_better']:.3f} |")
    print(f"\ntest set: {split_info['test']}; validation: {split_info['validation']}")
    print(f"done in {out['wall_s']}s")


if __name__ == "__main__":
    main()
