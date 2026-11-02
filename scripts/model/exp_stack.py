"""Cumulative feature stack: F6, then add one block at a time; each step is compared with the previous step (paired, match-cluster CI).

  ...cs2gpu python -I -X utf8 scripts\\model\\exp_stack.py --store combined --blocks hp,feat_x2_utility,feat_x1_kill_timing,feat_time \\
        --splits random,2024_to_2026,2026_to_2024 --seeds 0 [--extra-mono]
Blocks: "hp" (features_events.hp_features), "map" (one-hot), or the name of any feat_*.py module (compute(G, store)).
Writes data/<store>/exp_stack_<tag>.json
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import importlib
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import features_events as FE
import features_status as FS
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("--store", default="combined")
ap.add_argument("--tomes", default=None)
ap.add_argument("--blocks", default="hp")
ap.add_argument("--splits", default="random,2024_to_2026,2026_to_2024")
ap.add_argument("--seeds", default="0")
ap.add_argument("--extra-mono", action="store_true")
ap.add_argument("--tag", default="stack")
ap.add_argument("--stride", type=int, default=1, help="train/validate on every stride-th second")
ap.add_argument("--eras", default="", help="restrict to these eras, e.g. 2026 (the loadout features exist for 2026 only)")
args = ap.parse_args()
if args.extra_mono:
    os.environ["WP_EXTRA_MONO"] = "1"
tomes = args.tomes or args.store
G = W.load_grid(args.store)
if args.eras:
    G = G.filter(pl.col("era").is_in(args.eras.split(",")))
G = G.drop([c for c in ("label_source", "upload_day", "build_num", "score_lead_diag", "w_round_unused") if c in G.columns])
steps = [("F6", list(W.F6))]
cum = list(W.F6)
for blk in args.blocks.split(","):
    t0 = time.time()
    if blk == "hp":
        new = FE.hp_features(G, tomes)
    elif blk == "status":
        new = FS.loadout_features(G)
    elif blk == "map":
        new = pl.DataFrame({f"map_is_{m}": (G["map_name"] == f"de_{m}").cast(pl.Int8) for m in ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]})
    else:
        new = importlib.import_module(blk).compute(G, tomes)
    G = G.with_columns(new)
    cum = cum + new.columns
    steps.append((f"+{blk}", list(cum)))
    print(f"block {blk}: {len(new.columns)} columns ({time.time() - t0:.0f}s)", flush=True)
seeds = [int(s) for s in args.seeds.split(",")]
res = {}
for sname in args.splits.split(","):
    sp = W.make_split(sname, G, seed=0)
    te_sub, _, yte = W.arrays(G, sp["test"], W.F6)
    prev = None
    res[sname] = {}
    for label, feats in steps:
        _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=args.stride)
        _, Xva, yva = W.arrays(G, sp["val"], feats, stride=args.stride)
        _, Xte, _ = W.arrays(G, sp["test"], feats)
        ps = []
        for seed in seeds:
            m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=seed)
            ps.append(W.predict(m, best, Xte))
        p = np.mean(ps, axis=0)
        r = W.evaluate(te_sub, yte, p, ref=prev)
        entry = {"n_features": len(feats), "logloss": r["logloss"], "ci95": r["logloss_ci95"], "round_weighted": r["logloss_round_weighted"], "auc": r["auc"], "ece10": r["ece10"]}
        if prev is not None:
            entry["step_delta"] = r["delta_vs_ref"]
            entry["step_delta_ci95"] = r["delta_vs_ref_ci95"]
        base = W.evaluate(te_sub, yte, p, ref=res[sname]["F6"]["_p"]) if label != "F6" else None
        if base is not None:
            entry["total_delta_vs_F6"] = base["delta_vs_ref"]
            entry["total_delta_ci95"] = base["delta_vs_ref_ci95"]
        res[sname][label] = {**entry, "_p": p}
        d = f" step {entry['step_delta']:+.4f} total {entry['total_delta_vs_F6']:+.4f}" if prev is not None else ""
        print(f"{sname:<14} {label:<28} LL {r['logloss']:.4f} ({len(feats)} feats) AUC {r['auc']:.4f} ECE {r['ece10']:.4f}{d}", flush=True)
        prev = p
    for k in res[sname]:
        res[sname][k].pop("_p", None)
out = os.path.join(W.ROOT, "data", args.store, f"exp_{args.tag}.json")
json.dump({"features_final": steps[-1][1], "results": res, "extra_mono": args.extra_mono}, open(out, "w", encoding="utf-8"), indent=1)
print("->", out)
