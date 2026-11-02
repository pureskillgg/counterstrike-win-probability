"""Does an event-derived feature set add to the pure round state (F6)?  GPU, 3 seeds, match-level splits.

  ...python.exe -I -X utf8 scripts\\model\\exp_features.py --store combined --sets hp --splits random,2024_to_2026,2026_to_2024
Adds the new columns to F6 (alive-only monotone constraints), reports paired log-loss differences vs F6 with match-cluster CIs.
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

import features_events as FE
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--store", default="combined")
ap.add_argument("--tomes", default=None, help="tome store name (default: same as --store)")
ap.add_argument("--sets", default="hp")
ap.add_argument("--splits", default="random,2024_to_2026,2026_to_2024")
ap.add_argument("--seeds", default="0,1,2")
args = ap.parse_args()
tomes = args.tomes or args.store
G = W.load_grid(args.store)
new_cols = {}
if "hp" in args.sets:
    t0 = time.time()
    hp = FE.hp_features(G, tomes)
    G = G.with_columns(hp).with_columns((pl.col("hp_ct") - pl.col("hp_t")).alias("hp_diff"))
    new_cols["hp"] = ["hp_ct", "hp_t", "hp_diff"]
    print(f"hp features built in {time.time() - t0:.0f}s; mean hp_ct {G['hp_ct'].mean():.1f}  mean hp_t {G['hp_t'].mean():.1f}", flush=True)
    # consistency: with 5 alive and untouched, hp must be 500; with k alive, hp <= 100 * k
    bad = G.filter(pl.col("hp_ct") > 100 * pl.col("alive_ct") + 1e-3).height
    print(f"rows where hp_ct exceeds 100 x alive_ct: {bad} of {G.height}", flush=True)
if "map" in args.sets:
    MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
    G = G.with_columns([(pl.col("map_name") == f"de_{m}").cast(pl.Int8).alias(f"map_is_{m}") for m in MAPS])
    new_cols["map"] = [f"map_is_{m}" for m in MAPS]
res = {}
for sname in args.splits.split(","):
    sp = W.make_split(sname, G, seed=0)
    te_sub, _, yte = W.arrays(G, sp["test"], W.F6)
    res[sname] = {}
    ref = None
    for label, feats in [("F6", W.F6)] + [(f"F6+{k}", W.F6 + v) for k, v in new_cols.items()]:
        _, Xtr, ytr = W.arrays(G, sp["train"], feats)
        _, Xva, yva = W.arrays(G, sp["val"], feats)
        _, Xte, _ = W.arrays(G, sp["test"], feats)
        ps = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=seed)
            ps.append(W.predict(m, best, Xte))
        p = np.mean(ps, axis=0)
        if label == "F6":
            ref = p
        r = W.evaluate(te_sub, yte, p, ref=ref if label != "F6" else None)
        r["ll_by_seed"] = [float(W.ll_vec(q, yte).mean()) for q in ps]
        res[sname][label] = r
        d = f" delta {r['delta_vs_ref']:+.4f} [{r['delta_vs_ref_ci95'][0]:+.4f},{r['delta_vs_ref_ci95'][1]:+.4f}]" if "delta_vs_ref" in r else ""
        print(f"{sname:<14} {label:<10} LL {r['logloss']:.4f} (round-wtd {r['logloss_round_weighted']:.4f}) AUC {r['auc']:.4f} ECE {r['ece10']:.4f}{d}", flush=True)
out = os.path.join(W.ROOT, "data", args.store, f"exp_features_{args.sets.replace(',', '_')}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1)
print("->", out)
