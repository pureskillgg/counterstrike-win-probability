"""How many seconds per round does the model need? Train on n random seconds of every round, for n = 1, 2, 4, ... and on every second.   (XGBoost, GPU)

  ...python.exe -I -X utf8 scripts\\model\\exp_tick_sampling.py [--store combined] [--device cuda] [--seeds 2]

Motivation: the snapshot has one row per live second, and neighbouring seconds of a round are nearly the same state. The idea under test is to
take n random moments per round as the training rows, with n a hyperparameter. n = 1 is one random moment per round (the truly random
sample); n = 2 is twice the rows; and so on. Sampling is within each (match, round), so a round gives min(n, its live seconds) rows.
This is a proxy for finer-than-one-second ticks: it uses the seconds that already exist. It does not build tick-level snapshots.

Every model uses the same frozen split, the same 59 inputs and the same settings as the final model (depth 4, eta 0.05, min child weight 20,
subsample 0.8, monotone alive counts). Early stopping uses every validation second for every n, so only the training rows change.
Each model is scored on every second of the frozen test matches, paired against the every-second model with the same seed (the same seconds),
with a match-cluster 95% interval. Test predictions are saved to data/<store>/ckpt/ per cell with a fingerprint of the cell's settings
(split, features, parameters, n, seed); a rerun loads a cell only when the fingerprint matches.
Writes data/<store>/exp_tick_sampling.json.
"""
import argparse
import json
import os
import sys
import time

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl
import xgboost as xgb

import wp_common as W
from exp_pistol import BASE, Blocks, arrays

sys.stdout.reconfigure(encoding="utf-8")
N_LIST = [0, 1, 2, 4, 8, 16, 32, 64]  # 0 = every second; a live round is 65 seconds at the median (P5 32, P95 115)
KEYS = ["match_key", "round", "second"]


def sample_rows(base, n, seed):
    """Keys of n random seconds from every (match, round) of base."""
    rng = np.random.default_rng(seed)
    u = base.with_columns(pl.Series("u", rng.random(base.height)))
    u = u.with_columns(pl.col("u").rank("ordinal").over(["match_key", "round"]).alias("rk"))
    return u.filter(pl.col("rk") <= n).select(KEYS)


def ckpt_file(d, name):
    return os.path.join(d, "ckpt", f"{name}.npz")


def load_cell(path, fp):
    """The saved cell when it was made with the same settings; None otherwise."""
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if str(z["fp"]) != fp:
        print(f"{os.path.basename(path)}: checkpoint was made with other settings or split: training again", flush=True)
        return None
    return z


def fit(lf_tr, lf_va, tr, va, cols, mono, seed, device):
    dtr = xgb.QuantileDMatrix(Blocks(lf_tr, tr, cols, False, None, False), max_bin=256)
    dva = xgb.QuantileDMatrix(Blocks(lf_va, va, cols, False, None, False), ref=dtr)
    t = time.time()
    bst = xgb.train(dict(BASE, seed=seed, device=device, monotone_constraints=tuple(mono)), dtr, num_boost_round=3000,
                    evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    return bst, int(bst.best_iteration), int(dtr.num_row()), round(time.time() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    need = list(dict.fromkeys(["match_key", "round", "second", "label_ct_win", "w_round"] + feats))
    lf = pl.scan_parquet(os.path.join(d, "cache_snapshot_2026.parquet")).select(need)
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = list(W.mono_for(feats))

    base = lf.filter(pl.col("match_key").is_in(tr)).select(KEYS).collect()
    if base.is_duplicated().any():
        sys.exit("match/round/second is not unique in the training rows: sampling by second would be wrong")
    train_rounds = int(base.select(["match_key", "round"]).unique().height)
    Xte, yte, rte, mkte, wte = arrays(lf, te, feats, False)
    test_sub = pl.DataFrame({"match_key": mkte.to_list(), "w_round": wte.to_list()})
    print(f"train {len(tr)} matches, {base.height:,} live seconds, {train_rounds:,} rounds; test {len(te)} matches, {len(yte):,} seconds "
          f"({time.time() - t0:.0f}s)", flush=True)

    Xva, yva, _, mkva, wva = arrays(lf, va, feats, False)
    val_sub = pl.DataFrame({"match_key": mkva.to_list(), "w_round": wva.to_list()})
    preds, preds_va, info = {}, {}, {}
    for seed in range(a.seeds):
        for n in N_LIST:
            name = f"cell_n{n or 'all'}_seed{seed}_v2"  # v2: the checkpoints also carry the validation predictions
            fp = W.fingerprint("tick", feats, mono, BASE, tr, va, te, n, seed)
            z = load_cell(ckpt_file(d, name), fp)
            if z is not None:
                preds[(n, seed)] = z["p"]
                preds_va[(n, seed)] = z["pva"]
                info[(n, seed)] = {"train_rows": int(z["rows"]), "best_iteration": int(z["best"]), "fit_seconds": int(z["secs"])}
                print(f"{name}: loaded from checkpoint", flush=True)
                continue
            if n == 0:
                lf_tr = lf
            else:
                sel = sample_rows(base, n, seed)
                lf_tr = lf.join(sel.lazy(), on=KEYS, how="semi")
            bst, best, rows, secs = fit(lf_tr, lf, tr, va, feats, mono, seed, a.device)
            p = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, best + 1)).astype(np.float32)
            pva = bst.predict(xgb.DMatrix(Xva), iteration_range=(0, best + 1)).astype(np.float32)
            preds[(n, seed)] = p
            preds_va[(n, seed)] = pva
            info[(n, seed)] = {"train_rows": rows, "best_iteration": best, "fit_seconds": secs}
            os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
            tmp = os.path.join(d, "ckpt", f"{name}.tmp.npz")
            np.savez(tmp, p=p, pva=pva, rows=np.array(rows), best=np.array(best), secs=np.array(secs), fp=np.array(fp))
            os.replace(tmp, ckpt_file(d, name))
            print(f"{name}: {rows:,} training seconds, best iteration {best} ({secs}s, {time.time() - t0:.0f}s total)", flush=True)

    out = {"train_matches": len(tr), "validation_matches": len(va), "test_matches": len(te), "test_rows": int(len(yte)),
           "train_live_seconds": int(base.height), "train_rounds": train_rounds, "cells": {}, "summary": {}}
    for seed in range(a.seeds):
        for n in N_LIST:
            r = W.evaluate(test_sub, yte, preds[(n, seed)], ref=preds[(0, seed)])
            rv = W.evaluate(val_sub, yva, preds_va[(n, seed)], ref=preds_va[(0, seed)])
            out["cells"][f"n{n or 'all'}_seed{seed}"] = {
                "n_per_round": n or "all", "seed": seed, "train_rows": info[(n, seed)]["train_rows"],
                "best_iteration": info[(n, seed)]["best_iteration"], "fit_seconds": info[(n, seed)]["fit_seconds"],
                "logloss": r["logloss"], "logloss_ci95": r["logloss_ci95"], "logloss_round_weighted": r["logloss_round_weighted"],
                "auc": r["auc"], "ece10": r["ece10"], "delta_vs_every_second": r["delta_vs_ref"], "delta_vs_every_second_ci95": r["delta_vs_ref_ci95"],
                "validation_logloss": rv["logloss"], "validation_delta_vs_every_second": rv["delta_vs_ref"],
                "validation_delta_vs_every_second_ci95": rv["delta_vs_ref_ci95"]}
    for n in N_LIST:
        cells = [out["cells"][f"n{n or 'all'}_seed{s}"] for s in range(a.seeds)]
        out["summary"][f"n{n or 'all'}"] = {"train_rows": int(np.mean([c["train_rows"] for c in cells])),
                                            "mean_logloss": float(np.mean([c["logloss"] for c in cells])),
                                            "mean_delta_vs_every_second": float(np.mean([c["delta_vs_every_second"] for c in cells])),
                                            "mean_validation_logloss": float(np.mean([c["validation_logloss"] for c in cells])),
                                            "mean_validation_delta_vs_every_second": float(np.mean([c["validation_delta_vs_every_second"] for c in cells]))}
        s = out["summary"][f"n{n or 'all'}"]
        print(f"n={n or 'all':>4}: {s['train_rows']:>11,} training seconds  test log loss {s['mean_logloss']:.5f}  "
              f"delta vs every second {s['mean_delta_vs_every_second']:+.5f}  |  validation {s['mean_validation_logloss']:.5f}  "
              f"delta {s['mean_validation_delta_vs_every_second']:+.5f}", flush=True)
    W.dump_json(out, os.path.join(d, "exp_tick_sampling.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
