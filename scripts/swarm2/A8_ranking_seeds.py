"""A8 check 6: does the model ranking survive other split seeds? (primary split, seeds 0, 1, 2)

Calls bakeoff.fit_models (the bake-off's own fitting code: constant, LR, XGBoost, LightGBM, HistGB)
on the primary split for each seed. bakeoff.SEED is set per seed, so the split, the XGBoost and
LightGBM random states, and the HistGB internal split all follow the seed. Test log loss is reported
for each raw model (calibrated variants are not refitted here). Seed 0 must match
notes/model/bakeoff_results.md (XGBoost 0.43660, LightGBM 0.43760, HistGB 0.43691).
The test set is scored once per fit and not used for any choice.
Run from the project root:
  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A8_ranking_seeds.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [0, 1, 2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()
    xgbs = load("xgb_primary_seed0", ROOT / "scripts" / "model" / "xgb_primary_seed0.py")
    bo = xgbs.bo
    D = xgbs.build_data(dedupe=True)
    S, H, kept = D["S"], D["H"], D["kept_final"]
    orig = bo.SEED
    table = {}
    for seed in SEEDS:
        bo.SEED = seed
        sets = bo.make_splits(kept, H)[bo.PRIMARY]
        tr = S.filter(pl.col("match_key").is_in(sorted(sets["train"])))
        va = S.filter(pl.col("match_key").is_in(sorted(sets["val"])))
        te = S.filter(pl.col("match_key").is_in(sorted(sets["test"])))
        _, yte = bo.xy(te)
        res, _, _ = bo.fit_models(tr, va, te, np.zeros(va.height, dtype=int))
        _, yva = bo.xy(va)
        row = {}
        for name, r in res.items():
            if " + " in name:
                continue
            row[name] = (bo.metrics(yte, r["pte"])["logloss"], bo.metrics(yva, r["pva"])["logloss"])
        table[seed] = row
        print(f"seed {seed}: train {len(sets['train'])} val {len(sets['val'])} test {len(sets['test'])} matches; "
              f"time {time.time() - t0:.0f}s", flush=True)
        for name, (tl, vl) in row.items():
            print(f"  {name:<26} test LL {tl:.5f}   val LL {vl:.5f}", flush=True)
    bo.SEED = orig

    names = [n for n in table[SEEDS[0]].keys()]
    print("\n== test log loss by seed (raw models) ==")
    print(f"{'model':<26}" + "".join(f"{'seed ' + str(s):>12}" for s in SEEDS) + f"{'mean':>10}{'range':>10}")
    for n in names:
        vals = [table[s][n][0] for s in SEEDS]
        print(f"{n:<26}" + "".join(f"{v:>12.5f}" for v in vals) + f"{np.mean(vals):>10.5f}"
              f"{max(vals) - min(vals):>10.5f}")
    tree_names = ["XGBoost", "LightGBM", "HistGB"]
    print("\nbest tree model by test LL, per seed:")
    for s in SEEDS:
        best = min(tree_names, key=lambda n: table[s][n][0])
        order = sorted(tree_names, key=lambda n: table[s][n][0])
        print(f"  seed {s}: {best} best; order {order}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
