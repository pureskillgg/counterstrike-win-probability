"""Baselines on the combined per-second grid: constant, logistic on the alive difference, XGBoost on F6 / F9 / the retired
score-lead set, across four match-level splits, three seeds for XGBoost. GPU. Writes data/<store>/baselines.json.

  C:\\Users\\<user>\\.pixi\\workspace-envs\\cs2gpu-15282324412032665829\\envs\\default\\python.exe -I -X utf8 scripts\\model\\run_baselines.py [--store combined]
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"  # before numpy: the threaded BLAS crashes natively on this machine
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--store", default="combined")
ap.add_argument("--splits", default="random,2024_to_2026,2026_to_2024,chrono")
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
G = W.load_grid(args.store)
print("grid", G.shape, dict(G.group_by("era").len().iter_rows()), flush=True)
results = {}
t0 = time.time()
for sname in args.splits.split(","):
    sp = W.make_split(sname, G, seed=0)
    print(f"\n=== split {sname}: train {len(sp['train'])} val {len(sp['val'])} test {len(sp['test'])} matches", flush=True)
    te_sub, _, yte = W.arrays(G, sp["test"], W.F6)
    res = {}
    # constant
    _, _, ytr0 = W.arrays(G, sp["train"], W.F6)
    pc = np.full(len(yte), float(ytr0.mean()))
    res["constant"] = W.evaluate(te_sub, yte, pc)
    ref = None
    # logistic on the alive difference
    _, Xl, yl = W.arrays(G, sp["train"], ["ct_alive_minus_t"])
    lr = W.RidgeLogit().fit(Xl, yl)
    _, Xlt, _ = W.arrays(G, sp["test"], ["ct_alive_minus_t"])
    res["LR alive diff"] = W.evaluate(te_sub, yte, lr.predict(Xlt))
    for fname in ("F6", "F9", "F_SCORE"):
        feats = W.FEATURE_SETS[fname]
        _, Xtr, ytr = W.arrays(G, sp["train"], feats)
        _, Xva, yva = W.arrays(G, sp["val"], feats)
        _, Xte, _ = W.arrays(G, sp["test"], feats)
        ps = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=seed)
            ps.append(W.predict(m, best, Xte))
            print(f"  {fname} seed {seed}: best_iter {best}  test LL {float(W.ll_vec(ps[-1], yte).mean()):.4f}", flush=True)
        p = np.mean(ps, axis=0)
        if fname == "F6":
            ref = p
        res[f"XGB {fname} (mean of {len(ps)} seeds)"] = W.evaluate(te_sub, yte, p, ref=ref)
        res[f"XGB {fname} (mean of {len(ps)} seeds)"]["ll_by_seed"] = [float(W.ll_vec(q, yte).mean()) for q in ps]
        # per era on the test set
        for era in sorted(te_sub["era"].unique().to_list()):
            msk = (te_sub["era"] == era).to_numpy()
            res[f"XGB {fname} | era {era}"] = W.evaluate(te_sub.filter(pl.Series(msk)), yte[msk], p[msk])
    results[sname] = {"sizes": {k: len(v) for k, v in sp.items()}, "models": res}
    for k, v in res.items():
        print(f"  {k:<40} LL {v['logloss']:.4f} {v['logloss_ci95'][0]:.4f}-{v['logloss_ci95'][1]:.4f} | round-wtd {v['logloss_round_weighted']:.4f} | "
              f"AUC {v['auc']:.4f} ECE {v['ece10']:.4f} n={v['n_matches']}", flush=True)
out = os.path.join(W.ROOT, "data", args.store, "baselines.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
print(f"\ndone in {time.time() - t0:.0f}s -> {out}")
