"""Small input files for the foundation-model test, so the heavy PyTorch environments never load the 4.4-million-row cache.

  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\fm\\prep_fm_data.py [--n-ctx 60000] [--per-round 2] [--n-test 20000]

Writes data/<store>/fm/:
  context.npz   X (float32), y, match_index: up to --n-ctx labelled seconds from TRAINING matches only, at most --per-round seconds per round,
                in random order (the first n rows are a valid smaller context of size n)
  test.parquet  --n-test seconds drawn at random from the held-out TEST matches: match_key, round, second, w_round, label, p_xgb and the inputs;
                p_xgb = the final XGBoost settings (depth 4, eta 0.05, min child weight 20, monotone alive counts) fitted on the whole training set
                (stride 3, early stopping on the validation matches) on the CPU: the reference every foundation model is compared with, paired
Same split as exp_model_families.py (random match split, seed 0), same 52 inputs.
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--n-ctx", type=int, default=60000)
    ap.add_argument("--per-round", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = os.path.join(W.ROOT, "data", a.store, "fm")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(W.ROOT, "data", a.store, "cache_snapshot_2026.parquet")
    head = pl.scan_parquet(path).collect_schema().names()
    feats = W.F6 + [c for c in head if c.startswith("st_")]
    cols = list(dict.fromkeys(["match_key", "round", "second", "label_ct_win", "w_round", "era", "match_date", *feats]))  # era and match_date: make_split reads both
    G = pl.read_parquet(path, columns=cols)
    sp = W.make_split("random", G, seed=a.seed)
    print(f"{G.height:,} rows, {len(feats)} inputs; matches train/val/test {len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}", flush=True)

    rng = np.random.default_rng(a.seed)
    tr = G.filter(pl.col("match_key").is_in(sorted(sp["train"])))
    tr = tr.with_columns(pl.Series("_r", rng.random(tr.height))).sort("_r")
    tr = tr.with_columns(pl.int_range(pl.len()).over(["match_key", "round"]).alias("_k")).filter(pl.col("_k") < a.per_round).head(a.n_ctx)
    n_ctx = tr.height
    np.savez_compressed(os.path.join(out, "context.npz"), X=tr.select(feats).to_numpy().astype(np.float32), y=tr["label_ct_win"].to_numpy().astype(np.int64))
    print(f"context: {n_ctx:,} seconds from {tr['match_key'].n_unique()} training matches (CT-win rate {tr['label_ct_win'].mean():.3f})", flush=True)

    te_all = G.filter(pl.col("match_key").is_in(sorted(sp["test"])))
    te = te_all.sample(n=min(a.n_test, te_all.height), seed=a.seed)
    del te_all
    _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=3)
    _, Xva, yva = W.arrays(G, sp["val"], feats, stride=3)
    del G, tr
    t = time.time()
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cpu")
    p = W.predict(m, best, te.select(feats).to_numpy().astype(np.float32))
    te = te.with_columns(pl.Series("p_xgb", p))
    te.write_parquet(os.path.join(out, "test.parquet"))
    r = W.evaluate(te.select("match_key", "w_round", "label_ct_win"), te["label_ct_win"].to_numpy().astype(np.int64), p)
    json.dump({"features": feats, "n_context": int(n_ctx),"n_test": int(te.height), "xgb_logloss": r["logloss"], "xgb_ci95": r["logloss_ci95"],
               "xgb_best_iteration": best, "seed": a.seed}, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), indent=1)
    print(f"test: {te.height:,} seconds from {te['match_key'].n_unique()} held-out matches; XGBoost LL {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
