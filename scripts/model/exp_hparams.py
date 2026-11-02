"""Is the final XGBoost configuration still the right one on the larger store? (GPU sweep, paired, on the frozen split)

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\exp_hparams.py [--store combined] [--stride 3]

Same inputs as the final snapshot model (59), training matches of the frozen split, early stopping on the validation matches. Grid: tree depth {3, 4, 6, 8} x
min_child_weight {20, 100}; learning rate 0.05, 80% row subsample, monotone alive-count constraints as in the final model (depth 4, mcw 20 is the reference).
Every setting is scored on the SAME frozen TEST matches; the log-loss difference to the reference carries a match-cluster bootstrap interval. The test matches are
used only to report, never to choose: the setting is chosen on validation log loss. Writes data/<store>/exp_hparams.json.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    G = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"))
    sp = W.make_split("random", G, seed=0)
    feats = W.F6 + [c for c in G.columns if c.startswith("st_")] + [c for c in G.columns if c.startswith("map_is_")]
    feats = [f for f in dict.fromkeys(feats)]
    _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=a.stride)
    sva, Xva, yva = W.arrays(G, sp["val"], feats, stride=a.stride)
    ste, Xte, yte = W.arrays(G, sp["test"], feats, stride=1)
    print(f"{len(feats)} inputs; train {len(ytr):,} seconds (stride {a.stride}), val {len(yva):,}, test {len(yte):,} seconds in {len(sp['test'])} matches", flush=True)
    res, preds = [], {}
    for depth in (4, 3, 6, 8):
        for mcw in (20, 100):
            t0 = time.time()
            m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda", depth=depth, mcw=mcw)
            pva = W.predict(m, best, Xva)
            pte = W.predict(m, best, Xte)
            ll_va = float(-np.mean(yva * np.log(np.clip(pva, 1e-6, 1)) + (1 - yva) * np.log(np.clip(1 - pva, 1e-6, 1))))
            key = (depth, mcw)
            preds[key] = pte
            ref = preds.get((4, 20))
            r = W.evaluate(ste, yte, pte, ref=None if key == (4, 20) else ref)
            row = {"depth": depth, "min_child_weight": mcw, "best_iteration": int(best), "val_logloss": round(ll_va, 5), "test_logloss": r["logloss"], "test_logloss_ci95": r["logloss_ci95"],
                   "fit_seconds": round(time.time() - t0, 1)}
            if key != (4, 20):
                row["delta_vs_reference"] = r["delta_vs_ref"]
                row["delta_vs_reference_ci95"] = r["delta_vs_ref_ci95"]
            res.append(row)
            print(f"depth {depth} mcw {mcw}: val {ll_va:.5f} test {r['logloss']:.4f}" + (f" delta vs ref {r['delta_vs_ref']:+.5f} {[round(x, 5) for x in r['delta_vs_ref_ci95']]}" if key != (4, 20) else " (reference)") + f" iters {best} ({time.time() - t0:.0f}s)", flush=True)
    chosen = min(res, key=lambda r: r["val_logloss"])
    W.dump_json({"stride": a.stride, "inputs": len(feats), "test_matches": len(sp["test"]), "settings": res, "chosen_on_validation": {"depth": chosen["depth"], "min_child_weight": chosen["min_child_weight"]}},
                os.path.join(d, "exp_hparams.json"))
    print("chosen on validation:", chosen["depth"], chosen["min_child_weight"])


if __name__ == "__main__":
    main()
