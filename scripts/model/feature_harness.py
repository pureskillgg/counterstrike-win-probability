"""Evaluate one candidate feature module honestly: causality test, then paired gain over the pure round state (F6).

  ...python.exe -I -X utf8 scripts\\model\\feature_harness.py --module features_events --store combined_test [--seeds 0,1,2]
  (module = a file in scripts/model exposing compute(G, store) -> polars DataFrame of the new columns only)

Steps
  1. static guard: the module source may not read parquet itself (events only via features_events.scan_events), may not
     mention label_ct_win / final / winner / label_source / round_end, and may not read other rows' labels.
  2. causality test: 400 random grid rows (one per round, seed 0). For each, the module is recomputed on that row alone with
     every later event of the round hidden (features_events.CUTOFF). Any value that differs from the full computation by more
     than 1e-4 (relative 1e-6) fails the module; the first mismatches are printed.
  3. gain: XGBoost (GPU, alive-count monotone constraints, 3 seeds) on F6 and on F6 + new columns, same match-level splits;
     paired log-loss difference with a match-cluster 95% interval, per split, plus per-era numbers on the random split.
  Writes data/<store>/harness_<module>.json and prints a one-line verdict.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import importlib
import json
import re
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import features_events as FE
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
FORBIDDEN = [r"read_parquet", r"scan_parquet", r"glob\.", r"label_ct_win", r"label_source", r"\bwinner", r"round_end", r"\bfinal\b",
             r"rounds\.parquet", r"data[\\/]+combined"]


def static_guard(path):
    src = open(path, encoding="utf-8").read()
    # features_events itself is the sanctioned reader; only candidate modules are scanned
    if os.path.basename(path) == "features_events.py":
        return []
    return [p for p in FORBIDDEN if re.search(p, src)]


def causality_test(mod, G, store, n=400, seed=0):
    rows = (G.select("match_key", "round").with_row_index("_i").sample(fraction=1.0, shuffle=True, seed=seed)
            .unique(["match_key", "round"], keep="first", maintain_order=True).head(n))  # a random row from each of n rounds
    idx = sorted(rows["_i"].to_list())
    full = mod.compute(G, store)
    sub = G[idx]
    FE.CUTOFF = sub.select("match_key", "round", pl.col("tick").alias("cutoff_tick"))
    try:
        cut = mod.compute(sub, store)
    finally:
        FE.CUTOFF = None
    bad = []
    for c in full.columns:
        a = full[c].to_numpy()[idx].astype(float)
        b = cut[c].to_numpy().astype(float)
        both_nan = np.isnan(a) & np.isnan(b)
        diff = ~both_nan & ~np.isclose(a, b, rtol=1e-6, atol=1e-4, equal_nan=False)
        if diff.any():
            j = np.where(diff)[0][:3]
            bad.append((c, int(diff.sum()), [(float(a[k]), float(b[k])) for k in j]))
    return bad, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--store", default="combined_test")
    ap.add_argument("--tomes", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--splits", default="random,2024_to_2026,2026_to_2024")
    ap.add_argument("--eras", default="", help="restrict the experiment to these eras, e.g. 2026")
    args = ap.parse_args()
    tomes = args.tomes or args.store
    mod = importlib.import_module(args.module)
    path = os.path.join(os.path.dirname(__file__), args.module + ".py")
    verdict = {"module": args.module, "store": args.store}
    guard = static_guard(path)
    verdict["static_guard_hits"] = guard
    G = W.load_grid(args.store)
    if args.eras:
        G = G.filter(pl.col("era").is_in(args.eras.split(",")))
    t0 = time.time()
    bad, new = causality_test(mod, G, tomes)
    verdict["causality_failures"] = bad
    verdict["causality_seconds"] = round(time.time() - t0, 1)
    new_cols = new.columns
    verdict["new_columns"] = new_cols
    verdict["nan_share"] = {c: float(np.isnan(new[c].to_numpy().astype(float)).mean()) for c in new_cols}
    print(f"[{args.module}] static guard hits: {guard}; causality failures: {[(b[0], b[1]) for b in bad]}; new columns {new_cols}", flush=True)
    if guard or bad:
        verdict["verdict"] = "REJECTED (leakage guard)"
        out = os.path.join(W.ROOT, "data", args.store, f"harness_{args.module}.json")
        json.dump(verdict, open(out, "w", encoding="utf-8"), indent=1, default=str)
        print("REJECTED:", verdict["static_guard_hits"], verdict["causality_failures"])
        sys.exit(2)
    G = G.with_columns(new)
    res = {}
    for sname in args.splits.split(","):
        if args.eras and sname.startswith(("2024_to", "2026_to")):
            continue
        sp = W.make_split(sname, G, seed=0)
        te_sub, _, yte = W.arrays(G, sp["test"], W.F6)
        ps = {}
        for label, feats in (("F6", W.F6), ("F6+new", W.F6 + new_cols)):
            _, Xtr, ytr = W.arrays(G, sp["train"], feats)
            _, Xva, yva = W.arrays(G, sp["val"], feats)
            _, Xte, _ = W.arrays(G, sp["test"], feats)
            runs = []
            imp = None
            for seed in [int(s) for s in args.seeds.split(",")]:
                m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=seed)
                runs.append(W.predict(m, best, Xte))
                if label == "F6+new" and seed == 0:
                    g = m.get_booster().get_score(importance_type="gain")
                    imp = {feats[int(k[1:])]: round(v, 2) for k, v in g.items()}
            ps[label] = np.mean(runs, axis=0)
        r0 = W.evaluate(te_sub, yte, ps["F6"])
        r1 = W.evaluate(te_sub, yte, ps["F6+new"], ref=ps["F6"])
        entry = {"F6": r0["logloss"], "F6+new": r1["logloss"], "delta": r1["delta_vs_ref"], "delta_ci95": r1["delta_vs_ref_ci95"],
                 "round_weighted_F6": r0["logloss_round_weighted"], "round_weighted_new": r1["logloss_round_weighted"],
                 "ece_new": r1["ece10"], "n_matches": r1["n_matches"], "gain_importance": imp}
        if sname == "random":
            for era in sorted(te_sub["era"].unique().to_list()):
                msk = (te_sub["era"] == era).to_numpy()
                sub = te_sub.filter(pl.Series(msk))
                e1 = W.evaluate(sub, yte[msk], ps["F6+new"][msk], ref=ps["F6"][msk])
                entry[f"delta_era_{era}"] = [e1["delta_vs_ref"], e1["delta_vs_ref_ci95"]]
        res[sname] = entry
        print(f"[{args.module}] {sname:<14} F6 {r0['logloss']:.4f} -> {r1['logloss']:.4f}  delta {entry['delta']:+.4f} {entry['delta_ci95'][0]:+.4f}..{entry['delta_ci95'][1]:+.4f}", flush=True)
    verdict["results"] = res
    verdict["verdict"] = "ok"
    out = os.path.join(W.ROOT, "data", args.store, f"harness_{args.module}.json")
    json.dump(verdict, open(out, "w", encoding="utf-8"), indent=1, default=str)
    print("->", out)


if __name__ == "__main__":
    main()
