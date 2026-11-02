"""Random-tick sweep for the served 13-input model. For each round, n random ticks between freeze end and the effective end are used as
training rows, with n = 1, 2, 4, 8, 16, 32, 64 (nested: the first n of the 64 draws). Each cell trains the same XGBoost settings as
winprob_v4_basic (depth 4, eta 0.05, min child weight 20, subsample 0.8, early stopping after 100 rounds on the validation seconds,
monotone alive counts, GPU) and is scored on the test seconds, which are every second of the frozen test matches, the same rows as the
served model. Each cell is paired with the every-second model of the same seed (winprob_v4_basic for seed 0, winprob_v4_basic_s1 for seed 1).

Rows per second = n / mean round length (about 65 s), so n = 64 is about one row per second.

  ...python.exe -I -X utf8 scripts\\model\\exp_random_ticks.py [--seeds 0 1]
Writes data/combined/exp_random_ticks.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
import xgboost as xgb

import train_final as T
import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
NS = [1, 2, 4, 8, 16, 32, 64]
EVERY_SECOND_MODEL = {0: "winprob_v4_basic", 1: "winprob_v4_basic_s1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--store", default="combined")
    ap.add_argument("--prefix", default="random_tick_pool", help="pool file prefix: random_tick_pool (any tick) or random_second_pool (grid seconds, control)")
    ap.add_argument("--out", default="exp_random_ticks.json")
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()

    G, feats = T.load_basic(a.store)  # the served model's rows: every second, roster-defect rounds dropped
    sp = W.make_split("random", G, seed=0)
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    clean = G.select(["match_key", "round"]).unique()
    every_train_rows = int(G.filter(pl.col("match_key").is_in(tr)).height)
    sub_va, Xva, yva = W.arrays(G, va, feats, stride=1)
    sub_te, Xte, yte = W.arrays(G, te, feats, stride=1)
    mono = list(W.mono_for(feats))
    print(f"every-second train rows {every_train_rows:,}; validation {len(yva):,}; test {len(yte):,} ({time.time() - t0:.0f}s)", flush=True)

    out = {"n_list": NS, "draws_per_round": 64, "every_second_train_rows": every_train_rows, "cells": {}}
    trset = set(tr)
    for seed in a.seeds:
        pool = pl.read_parquet(os.path.join(d, f"{a.prefix}_s{seed}.parquet"))
        pool = pool.filter(pl.col("match_key").is_in(list(trset))).join(clean, on=["match_key", "round"], how="semi")
        ref_name = EVERY_SECOND_MODEL[seed]
        ref_meta = json.load(open(os.path.join(W.ROOT, "models", f"{ref_name}.meta.json"), encoding="utf-8"))
        ref = xgb.Booster()
        ref.load_model(os.path.join(W.ROOT, "models", f"{ref_name}.json"))
        best_ref = int(ref_meta["best_iteration"])
        p_ref = ref.predict(xgb.DMatrix(Xte), iteration_range=(0, best_ref + 1))
        p_ref_va = ref.predict(xgb.DMatrix(Xva), iteration_range=(0, best_ref + 1))
        print(f"seed {seed}: pool train rows {pool.height:,} ({time.time() - t0:.0f}s)", flush=True)

        for n in NS:
            sub = pool.filter(pl.col("draw_rank") < n)
            Xtr = sub.select(feats).to_numpy().astype("float32")
            ytr = sub["label_ct_win"].to_numpy()
            m, best = W.fit_xgb(Xtr, ytr, Xva, yva, mono, seed=seed, device="cuda")
            p = W.predict(m, best, Xte)
            pva = W.predict(m, best, Xva)
            ev = W.evaluate(sub_te, yte, p, ref=p_ref)
            evv = W.evaluate(sub_va, yva, pva, ref=p_ref_va)
            cell = {
                "seed": seed, "n_ticks_per_round": n, "train_rows": int(len(ytr)),
                "rows_vs_every_second": round(len(ytr) / every_train_rows, 4),
                "best_iteration": int(best), "validation_logloss": evv["logloss"],
                "test_logloss": ev["logloss"], "test_logloss_ci95": ev["logloss_ci95"],
                "test_delta_vs_every_second": ev["delta_vs_ref"], "test_delta_ci95": ev["delta_vs_ref_ci95"],
                "validation_delta_vs_every_second": evv["delta_vs_ref"], "validation_delta_ci95": evv["delta_vs_ref_ci95"],
            }
            out["cells"][f"s{seed}_n{n}"] = cell
            print(f"seed {seed} n={n:>2}: rows {len(ytr):>10,} ({cell['rows_vs_every_second']:.3f} of every second) best {best} "
                  f"test {ev['logloss']:.5f} delta {ev['delta_vs_ref']:+.5f} [{ev['delta_vs_ref_ci95'][0]:+.5f}, {ev['delta_vs_ref_ci95'][1]:+.5f}] "
                  f"validation {evv['logloss']:.5f} ({time.time() - t0:.0f}s)", flush=True)
            W.dump_json(out, os.path.join(d, "exp_random_ticks.json"))
        del pool
    out["reference_test_logloss"] = {f"seed{s}": json.load(open(os.path.join(W.ROOT, "models", f"{EVERY_SECOND_MODEL[s]}.meta.json"),
                                                                   encoding="utf-8"))["test_logloss"] for s in a.seeds}
    W.dump_json(out, os.path.join(d, "exp_random_ticks.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
