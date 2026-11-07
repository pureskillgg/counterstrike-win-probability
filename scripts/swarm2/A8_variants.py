"""A8 check 3: split-seed sensitivity and the round-end-tick (post-decision) states.

Variants, primary split, split seeds 0, 1, 2 (XGBoost settings as in bakeoff.fit_models):
  V0  baseline: all states (includes states at the round_end tick).
  V1  round-end-tick states removed from train, validation and test: a state is dropped when its
      tick equals the round's round_end tick (the death that ends the round is then not in any
      state). Every round keeps its live-start state. ASSUMPTION: states at the round_end tick are
      the post-decision states; the decided-state check in A8_leakage_check.py supports this.
For V0 the test set is also scored on non-round-end states, so V0 and V1 can be compared on the
same states. Seed 0 V0 must reproduce the reported test log loss 0.43660 (within 0.0002).
Early stopping uses the validation split only. Test is scored once per fit and never used to pick
iterations. Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_variants.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_eval(bo, S, sets, seed, eval_mask_te=None):
    tr = S.filter(pl.col("match_key").is_in(sorted(sets["train"])))
    va = S.filter(pl.col("match_key").is_in(sorted(sets["val"])))
    te = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
    Xtr, ytr = bo.xy(tr)
    Xva, yva = bo.xy(va)
    Xte, yte = bo.xy(te)
    mX = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=4, min_child_weight=20,
                           subsample=0.8, tree_method="hist", device="cpu",
                           monotone_constraints=tuple(bo.MONO), early_stopping_rounds=100,
                           eval_metric="logloss", random_state=seed)
    t0 = time.time()
    mX.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    bx = int(mX.best_iteration)
    pva = mX.predict_proba(Xva, iteration_range=(0, bx + 1))[:, 1]
    pte = mX.predict_proba(Xte, iteration_range=(0, bx + 1))[:, 1]
    out = {"seed": seed, "best_iteration": bx, "fit_s": round(time.time() - t0, 1),
           "states": {"train": tr.height, "val": va.height, "test": te.height},
           "val": bo.metrics(yva, pva), "test": bo.metrics(yte, pte)}
    if eval_mask_te is not None:
        m = eval_mask_te
        out["test_non_end"] = bo.metrics(yte[m], pte[m])
        out["test_end"] = bo.metrics(yte[~m], pte[~m])
    return out


def show(tag, r):
    t = r["test"]
    print(f"  {tag}: seed {r['seed']} best_iteration {r['best_iteration']} fit {r['fit_s']}s "
          f"states train/val/test {r['states']['train']}/{r['states']['val']}/{r['states']['test']}")
    print(f"      val LL {r['val']['logloss']:.5f}; test (all eval states, n={t['n']}) LL {t['logloss']:.5f} "
          f"Brier {t['brier']:.5f} AUC {t['auc']:.4f} ECE {t['ece']:.4f} acc {t['acc']:.4f}")
    if "test_non_end" in r:
        a = r["test_non_end"]
        b = r["test_end"]
        print(f"      test on non-round-end states (n={a['n']}) LL {a['logloss']:.5f} Brier {a['brier']:.5f} "
              f"AUC {a['auc']:.4f} ECE {a['ece']:.4f} acc {a['acc']:.4f}")
        print(f"      test on round-end-tick states (n={b['n']}) LL {b['logloss']:.5f} AUC {b['auc']:.4f} "
              f"acc {b['acc']:.4f}")
    sys.stdout.flush()


def main():
    t0 = time.time()
    xgbs = load("xgb_primary_seed0", ROOT / "scripts" / "model" / "xgb_primary_seed0.py")
    bo = xgbs.bo
    fv = bo.fv
    print("== A8 variants: round-end-tick states and split seed ==")
    D = xgbs.build_data(dedupe=True)
    S = D["S"]
    H = D["H"]
    kept = D["kept_final"]
    RE = fv.read("round_end", ["match_key", "round", "tick"])
    ends = RE.select(["match_key", pl.col("round").cast(pl.Int64), pl.col("tick").alias("end_tick")])
    S = S.with_columns(pl.col("round").cast(pl.Int64)).join(ends, on=["match_key", "round"], how="left")
    assert S["end_tick"].null_count() == 0, "a state has no round_end row"
    S = S.with_columns((pl.col("tick") == pl.col("end_tick")).alias("is_end"))
    n_end = int(S["is_end"].sum())
    print(f"states {S.height}; states at round_end tick {n_end} ({100 * n_end / S.height:.2f}%)")
    S_v1 = S.filter(~pl.col("is_end"))
    print(f"V1 states after removing round-end-tick states: {S_v1.height}; "
          f"every round keeps its live start: "
          f"{S_v1.filter(pl.col('kind') == 'live_start').height} live-start states")

    orig_seed = bo.SEED
    results = {"V0": [], "V1": []}
    for seed in SEEDS:
        bo.SEED = seed
        sets = bo.make_splits(kept, H)[bo.PRIMARY]
        bo.SEED = orig_seed if seed != 0 else seed
        if seed == 0:
            ref = D["sets"]
            same = all(ref[p] == sets[p] for p in ("train", "val", "test"))
            print(f"\nseed 0 split equals bake-off primary split: {same} "
                  f"(train {len(sets['train'])}, val {len(sets['val'])}, test {len(sets['test'])})")
        else:
            print(f"\nseed {seed} split: train {len(sets['train'])}, val {len(sets['val'])}, "
                  f"test {len(sets['test'])} matches")
        # V0: all states; also score test on non-end states
        te_v0 = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
        m_te = (~te_v0["is_end"]).to_numpy()
        r0 = fit_eval(bo, S, sets, seed, eval_mask_te=m_te)
        results["V0"].append(r0)
        show("V0 all states", r0)
        if seed == 0:
            ok = abs(r0["test"]["logloss"] - 0.43660) <= 0.0002
            print(f"      seed-0 V0 reproduces reported test LL 0.43660 within 0.0002: {ok}")
        # V1: round_end-tick states removed everywhere; eval on remaining test states
        r1 = fit_eval(bo, S_v1, sets, seed)
        results["V1"].append(r1)
        show("V1 without round-end states", r1)
        bo.SEED = orig_seed
        sys.stdout.flush()

    print("\n== summary ==")
    for name in ("V0", "V1"):
        lls = [r["test"]["logloss"] for r in results[name]]
        aucs = [r["test"]["auc"] for r in results[name]]
        bis = [r["best_iteration"] for r in results[name]]
        print(f"{name}: test LL by seed {[round(x, 5) for x in lls]} mean {np.mean(lls):.5f} "
              f"range {max(lls) - min(lls):.5f}; test AUC by seed {[round(x, 4) for x in aucs]}; "
              f"best_iteration {bis}")
    nonend0 = [r["test_non_end"]["logloss"] for r in results["V0"]]
    print(f"V0 evaluated on non-round-end states: LL by seed {[round(x, 5) for x in nonend0]} "
          f"mean {np.mean(nonend0):.5f}")
    v1 = [r["test"]["logloss"] for r in results["V1"]]
    print(f"V1 on its own (non-round-end) test states: LL by seed {[round(x, 5) for x in v1]} "
          f"mean {np.mean(v1):.5f}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
