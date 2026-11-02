"""Does a model trained on EARLIER uploads still work on the NEWEST uploads? (GPU, on the cached 2026 snapshot table.)

  ...cs2gpu python.exe -I -X utf8 scripts\\model\\exp_temporal.py [--cutoff 2026/10/10] [--stride 3]

Two models with identical settings and inputs (the 52 snapshot inputs, no map input, training on every stride-th second):
  reference : trained on the frozen-split TRAINING matches of every upload day (what the final model does)
  temporal  : trained only on the training / validation matches uploaded on or before the cutoff day
Both are scored on the frozen-split TEST matches uploaded after the cutoff, a set neither model saw, with a paired match-cluster
interval for the difference. The difference is the price of not having seen the newest days, i.e. how much a model that is a few days
old loses on new matches. A second line scores the temporal model on ALL matches uploaded after the cutoff (those are all unseen by it).
Writes data/<store>/exp_temporal.json.
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
    ap.add_argument("--cutoff", default="2026/10/10", help="last upload day the temporal model may train on (as in matches.parquet upload_day)")
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    G = pl.read_parquet(os.path.join(d, "cache_snapshot_2026.parquet"))
    M = pl.read_parquet(os.path.join(d, "matches.parquet")).select("match_key", "upload_day")
    day = dict(M.iter_rows())
    sp = W.make_split("random", G, seed=0)
    ST = [c for c in G.columns if c.startswith("st_")]
    feats = W.F6 + ST
    early = {k for k in set(sp["train"]) | set(sp["val"]) if day.get(k, "9999") <= a.cutoff}
    tr_ref, va_ref = sp["train"], sp["val"]
    tr_tmp = {k for k in sp["train"] if k in early}
    va_tmp = {k for k in sp["val"] if k in early}
    late_all = {k for k in G["match_key"].unique().to_list() if day.get(k, "0") > a.cutoff}
    late_test = {k for k in sp["test"] if k in late_all}
    print(f"cutoff {a.cutoff}: reference trains on {len(tr_ref)} matches, temporal on {len(tr_tmp)}; later-day matches {len(late_all)} (frozen test among them {len(late_test)})", flush=True)
    if len(late_test) < 50 or len(tr_tmp) < 500:
        raise SystemExit("cutoff leaves too few matches on one side")

    def fit(trk, vak, tag):
        t0 = time.time()
        _, Xtr, ytr = W.arrays(G, trk, feats, stride=a.stride)
        _, Xva, yva = W.arrays(G, vak, feats, stride=a.stride)
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda")
        print(f"{tag}: {len(ytr):,} training seconds, best iteration {best} ({time.time() - t0:.0f}s)", flush=True)
        return m, best

    ref, b_ref = fit(tr_ref, va_ref, "reference")
    tmp, b_tmp = fit(tr_tmp, va_tmp, "temporal")
    out = {"cutoff": a.cutoff, "train_matches": {"reference": len(tr_ref), "temporal": len(tr_tmp)}, "later_day_matches": len(late_all),
           "later_day_test_matches": len(late_test), "stride": a.stride}
    sub, X, y = W.arrays(G, late_test, feats)
    p_ref, p_tmp = W.predict(ref, b_ref, X), W.predict(tmp, b_tmp, X)
    r_ref = W.evaluate(sub, y, p_ref)
    r_tmp = W.evaluate(sub, y, p_tmp, ref=p_ref)
    keep = ("logloss", "logloss_ci95", "auc", "ece10", "ece10_ci95", "smooth_ece")
    out["frozen_test_matches_after_cutoff"] = {"n_matches": r_ref["n_matches"], "n_rows": r_ref["n_rows"], "reference": {k: r_ref[k] for k in keep},
                                              "temporal": {k: r_tmp[k] for k in keep}, "temporal_minus_reference": r_tmp["delta_vs_ref"], "temporal_minus_reference_ci95": r_tmp["delta_vs_ref_ci95"]}
    sub2, X2, y2 = W.arrays(G, late_all, feats)
    r2 = W.evaluate(sub2, y2, W.predict(tmp, b_tmp, X2))
    out["temporal_on_all_later_matches"] = {"n_matches": r2["n_matches"], "n_rows": r2["n_rows"], **{k: r2[k] for k in keep}}
    W.dump_json(out, os.path.join(d, "exp_temporal.json"))
    f = out["frozen_test_matches_after_cutoff"]
    print(f"later-day test matches {f['n_matches']}: reference {f['reference']['logloss']:.4f}, temporal {f['temporal']['logloss']:.4f}; temporal minus reference "
          f"{f['temporal_minus_reference']:+.4f} {[round(x, 4) for x in f['temporal_minus_reference_ci95']]}", flush=True)


if __name__ == "__main__":
    main()
