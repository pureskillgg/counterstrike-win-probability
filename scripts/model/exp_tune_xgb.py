"""GPU random search over the XGBoost settings of the final model, selected on VALIDATION matches only.

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\exp_tune_xgb.py [--trials 40] [--stride 3]

Same rows, inputs, split (random match split, seed 0) and monotone constraints as exp_model_families.py. Each trial fits on the training matches
(every `stride`-th second) with early stopping on the validation matches; the trial's score is its validation log loss. The configuration
with the best validation score is then scored on the TEST matches, paired against the current settings (depth 4, eta 0.05, min child weight
20, subsample 0.8), with a match-cluster interval. The test matches are never used to choose anything. Writes data/<store>/exp_tune_xgb.json
after every trial, so an interrupted run keeps its results (a rerun skips finished trials).
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
BASELINE = {"depth": 4, "eta": 0.05, "mcw": 20, "subsample": 0.8}
DEVICE = "cpu"  # set from --device in main()


def sample(rng):
    return {"depth": int(rng.choice([3, 4, 5, 6, 7, 8])), "eta": float(rng.choice([0.03, 0.05, 0.08])), "mcw": int(rng.choice([5, 10, 20, 50, 100, 200])),
            "subsample": float(rng.choice([0.6, 0.8, 1.0])), "colsample_bytree": float(rng.choice([0.5, 0.7, 1.0])),
            "reg_lambda": float(rng.choice([0.5, 1.0, 3.0, 10.0])), "gamma": float(rng.choice([0.0, 0.5, 2.0]))}


def fit(cfg, Xtr, ytr, Xva, yva, mono):
    c = dict(cfg)
    depth, eta, mcw = c.pop("depth"), c.pop("eta"), c.pop("mcw")
    return W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=0, device=DEVICE, depth=depth, eta=eta, mcw=mcw, **c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device
    out_path = os.path.join(W.ROOT, "data", args.store, "exp_tune_xgb.json")
    cache = os.path.join(W.ROOT, "data", args.store, "cache_snapshot_2026.parquet")
    signature = {"stride": args.stride, "cache_bytes": os.path.getsize(cache), "cache_mtime": int(os.path.getmtime(cache))}  # trials are comparable only on the same data and stride
    res = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {"trials": []}
    if res.get("signature") not in (None, signature):
        sys.exit(f"{out_path} holds trials run with {res.get('signature')}, this run has {signature}: move the old file away to start a new search")
    res["signature"] = signature
    G = pl.read_parquet(cache)
    sp = W.make_split("random", G, seed=0)
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]  # the 59 inputs of the final model
    mono = W.mono_for(feats)
    _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=args.stride)
    _, Xva, yva = W.arrays(G, sp["val"], feats, stride=args.stride)
    te_sub, Xte, yte = W.arrays(G, sp["test"], feats)
    print(f"train {len(ytr):,} val {len(yva):,} test {len(yte):,} rows; {len(feats)} features", flush=True)

    def val_ll(m, best):
        p = np.clip(W.predict(m, best, Xva), 1e-15, 1 - 1e-15)
        return float(-np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p)))

    rng = np.random.default_rng(123)
    cfgs = [dict(BASELINE)] + [sample(rng) for _ in range(args.trials)]
    done = {json.dumps(t["config"], sort_keys=True) for t in res["trials"]}
    for i, cfg in enumerate(cfgs):
        if json.dumps(cfg, sort_keys=True) in done:
            continue
        t0 = time.time()
        m, best = fit(cfg, Xtr, ytr, Xva, yva, mono)
        v = val_ll(m, best)
        res["trials"].append({"config": cfg, "val_logloss": v, "best_iteration": best, "seconds": time.time() - t0, "baseline": i == 0})
        W.dump_json(res, out_path)
        print(f"trial {i:>2}: val {v:.5f} best_it {best:>4} {time.time() - t0:>4.0f}s {cfg}", flush=True)
    trials = sorted(res["trials"], key=lambda t: t["val_logloss"])
    base = next(t for t in res["trials"] if t["baseline"])
    top = [t for t in trials if not t["baseline"]][:3]
    mb, bb = fit(base["config"], Xtr, ytr, Xva, yva, mono)
    p_base = W.predict(mb, bb, Xte)
    r_base = W.evaluate(te_sub, yte, p_base)
    res["test"] = {"baseline": {"config": base["config"], "val_logloss": base["val_logloss"], "logloss": r_base["logloss"], "ci95": r_base["logloss_ci95"], "auc": r_base["auc"]}, "selected_by_validation": []}
    for t in top:
        m, b = fit(t["config"], Xtr, ytr, Xva, yva, mono)
        r = W.evaluate(te_sub, yte, W.predict(m, b, Xte), ref=p_base)
        res["test"]["selected_by_validation"].append({"config": t["config"], "val_logloss": t["val_logloss"], "logloss": r["logloss"], "ci95": r["logloss_ci95"], "auc": r["auc"],
                                                      "delta_vs_baseline": r["delta_vs_ref"], "delta_ci95": r["delta_vs_ref_ci95"]})
        print(f"test: val {t['val_logloss']:.5f} -> test {r['logloss']:.4f} (baseline {r_base['logloss']:.4f}) delta {r['delta_vs_ref']:+.5f} {[round(x, 5) for x in r['delta_vs_ref_ci95']]} {t['config']}", flush=True)
    res["n_trials"] = len(res["trials"]) - 1
    W.dump_json(res, out_path)
    print("->", out_path)


if __name__ == "__main__":
    main()
