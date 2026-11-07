"""Small training/validation arrays for the fine-tuning runs, so the PyTorch processes never load the 4.4-million-row cache (that cost them 6-10 GB of host memory
and got them killed when other jobs were using the machine). This script itself keeps its peak memory low: it picks the rows from three narrow columns first and
then reads only the chosen rows' inputs.

  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\fm\\prep_ft_data.py [--per-round 3] [--val-per-round 2] [--val-rows 12000]

Writes data/<store>/fm/ft_data.npz: Xtr, ytr (up to --per-round seconds per training round, TRAINING matches only, random order), Xva, yva (up to --val-per-round seconds
per round from the VALIDATION matches, at most --val-rows rows), and feats. Same split as everything else (random match split, seed 0, stratified by era).
"""
import os

os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
import argparse
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")


def pick(lf, keys, feats, per_round, seed, cap=None):
    rng = np.random.default_rng(seed)
    idx = (lf.select("match_key", "round", "second").filter(pl.col("match_key").is_in(sorted(keys))).collect())
    idx = idx.with_columns(pl.Series("_r", rng.random(idx.height))).sort("_r")
    idx = idx.with_columns(pl.int_range(pl.len()).over(["match_key", "round"]).alias("_k")).filter(pl.col("_k") < per_round).select("match_key", "round", "second")
    sel = lf.join(idx.lazy(), on=["match_key", "round", "second"], how="inner").select(*feats, "label_ct_win").collect(engine="streaming")
    perm = np.random.default_rng(seed + 1).permutation(sel.height)
    if cap:
        perm = perm[:cap]
    return sel.select(feats).to_numpy().astype(np.float32)[perm], sel["label_ct_win"].to_numpy().astype(np.int64)[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--per-round", type=int, default=3)
    ap.add_argument("--val-per-round", type=int, default=2)
    ap.add_argument("--val-rows", type=int, default=12000)
    a = ap.parse_args()
    base = os.path.join(W.ROOT, "data", a.store)
    meta = json.load(open(os.path.join(base, "fm", "meta.json"), encoding="utf-8"))
    feats = meta["features"]
    lf = pl.scan_parquet(os.path.join(base, "cache_snapshot_2026.parquet"))
    G = lf.select("match_key", "era", "match_date").unique("match_key").collect()  # make_split reads match_key, era, match_date (one row per match is enough)
    sp = W.make_split("random", G, seed=meta.get("seed", 0))
    Xtr, ytr = pick(lf, sp["train"], feats, a.per_round, 1)
    Xva, yva = pick(lf, sp["val"], feats, a.val_per_round, 2, cap=a.val_rows)
    np.savez_compressed(os.path.join(base, "fm", "ft_data.npz"), Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, feats=np.array(feats))
    print(f"train {len(ytr):,} rows from {len(sp['train'])} training matches; validation {len(yva):,} rows from {len(sp['val'])} validation matches -> {os.path.join(base, 'fm', 'ft_data.npz')}")


if __name__ == "__main__":
    main()
