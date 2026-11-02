"""Seed averaging for the best measured feature set: 58 inputs (59 minus heavy, plus score lead), baseline settings.

Five XGBoost seeds (0 to 4), each with early stopping on validation. The averaged prediction is the mean of the five probabilities.
Reports each seed's validation and test log loss, and the averaged model's log loss paired with seed 0 on the same seconds.
Each seed's booster is saved as models/winprob_v8_seed<k>.json; the metadata records the split hash and the averaged scores (served: false).

  ...python.exe -I -X utf8 scripts\\model\\exp_seed_avg58.py
Writes data/combined/exp_seed_avg58.json.
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
HEAVY = ["st_heavy_ct", "st_heavy_t"]
SEEDS = [0, 1, 2, 3, 4]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    t0 = time.time()
    d = os.path.join(W.ROOT, "data", "combined")
    ref_meta = json.load(open(os.path.join(W.ROOT, "models", "winprob_v4.meta.json"), encoding="utf-8"))
    feats59 = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    mono59 = {f: int(ref_meta["monotone_constraints"][f]) for f in feats59}
    feats = [f for f in feats59 if f not in HEAVY] + ["score_lead_diag"]
    mono = [mono59.get(f, 0) for f in feats]

    need = list(dict.fromkeys(["match_key", "round", "tick", "second", "label_ct_win", "w_round"] + feats59))
    G = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need).with_columns(pl.col(feats59).cast(pl.Float32)).collect()
    lead = (pl.scan_parquet(os.path.join(d, "grid.parquet")).filter(pl.col("era") == "2026").select(["match_key", "round", "score_lead_diag"])
            .unique(subset=["match_key", "round"]).collect())
    G = G.join(lead, on=["match_key", "round"], how="left").with_columns(pl.col("score_lead_diag").fill_null(0).cast(pl.Float32))
    kdf = G.select("match_key").unique().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                      pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    del kdf
    _, Xtr, ytr = W.arrays(G, tr, feats, stride=1)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)

    out = {"features": len(feats), "seeds": [], "split_manifest_sha256": sha(os.path.join(d, "split_manifest.json"))}
    pva_all, pte_all = [], []
    for seed in SEEDS:
        m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=seed, device="cuda")
        pva, pte = W.predict(m, best, Xva), W.predict(m, best, Xte)
        pva_all.append(pva)
        pte_all.append(pte)
        ll_va = W.evaluate(sub_va, yva, pva)["logloss"]
        ll_te = W.evaluate(sub_te, yte, pte)["logloss"]
        out["seeds"].append({"seed": seed, "best_iteration": int(best), "validation_logloss": ll_va, "test_logloss": ll_te})
        m.get_booster().save_model(os.path.join(W.ROOT, "models", f"winprob_v8_seed{seed}.json"))
        print(f"seed {seed}: best iteration {best}; validation {ll_va:.6f}; test {ll_te:.6f} ({time.time() - t0:.0f}s)", flush=True)
        W.dump_json(out, os.path.join(d, "exp_seed_avg58.json"))

    pva_mean = np.mean(pva_all, axis=0)
    pte_mean = np.mean(pte_all, axis=0)
    ev_va = W.evaluate(sub_va, yva, pva_mean)
    ev_te = W.evaluate(sub_te, yte, pte_mean, ref=pte_all[0])
    out["averaged"] = {"validation_logloss": ev_va["logloss"], "test_logloss": ev_te["logloss"], "test_rows": int(len(yte)),
                       "test_delta_vs_seed0": ev_te["delta_vs_ref"], "test_delta_vs_seed0_ci95": ev_te["delta_vs_ref_ci95"]}
    out["seed_mean_test_logloss"] = float(np.mean([s["test_logloss"] for s in out["seeds"]]))
    W.dump_json(out, os.path.join(d, "exp_seed_avg58.json"))
    lo, hi = ev_te["delta_vs_ref_ci95"]
    print(f"averaged 5 seeds: validation {ev_va['logloss']:.6f}; test {ev_te['logloss']:.6f} "
          f"(vs seed 0 {ev_te['delta_vs_ref']:+.5f} [{lo:+.5f}, {hi:+.5f}]); mean of single seeds {out['seed_mean_test_logloss']:.6f} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
