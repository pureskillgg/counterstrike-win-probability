"""Positions model, validation only: no test-split or holdout rows enter a model, an early stop or a score.

Fit on the training matches minus an early-stopping set (10% of the training matches, chosen by crc32 of the key), stop on that set,
and score the validation matches at every second the position table has. The frozen test split and holdout B are never selected.

    CS2_ROOT=G:/pureskill/csds-cs2 python -I -X utf8 positions_validation.py --pos <cache_pos.parquet> --features full|base \
        --seed S --device cuda --out <preds.parquet>

Writes match_key, round, second, label_ct_win and p (the validation predictions). The early-stopping set and the fit set are printed.
"""
import argparse
import json
import os
import sys
import zlib

ROOT = os.environ.get("CS2_ROOT", "G:/pureskill/csds-cs2")
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import wp_common as W  # noqa: E402

GEO = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_in_contact", "t_in_contact", "c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist", "defuse_slack"]
VIEW = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
BLIND = ["c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]
POS_KEY = ["match_key", "round", "tick"]
TRAIN_CACHE = os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet")
HOLDOUT_B_KEYS = os.path.join(ROOT, "data", "combined", "holdout_22_25_test_keys.json")
ES_SHARE = 10  # percent of the training matches held out for early stopping


def to_xy(df, feats):
    X = df.select(feats).to_numpy()
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def early_stop_set(keys):
    return {k for k in keys if zlib.crc32(f"es:{k}".encode()) % 100 < ES_SHARE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--features", choices=["full", "base"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = list(pl.read_parquet_schema(TRAIN_CACHE).keys())
    cache_base = [c for c in W.F6 if c in names] + [c for c in names if c.startswith("map_is_")] + [c for c in names if c.startswith("st_")]
    base = list(dict.fromkeys(cache_base))
    feats = base if a.features == "base" else base + GEO + VIEW + BLIND
    pos_feats = [c for c in feats if c not in names]
    print(f"features {a.features}: {len(feats)} inputs ({len(pos_feats)} from {a.pos})", flush=True)
    keep_cols = ["second"] if "second" in names else []

    pos = (pl.scan_parquet(a.pos).select(POS_KEY + pos_feats)
           .with_columns([pl.col(c).cast(pl.Float32) for c in pos_feats]))

    meta = pl.scan_parquet(TRAIN_CACHE).select(["match_key", "era", "match_date"]).unique(subset=["match_key"]).collect()
    sp = W.make_split("random", meta, seed=0)
    es = early_stop_set(sp["train"])
    fit = sp["train"] - es
    val = sp["val"]
    assert not (val & sp["test"]) and not (fit & sp["test"]) and not (es & sp["test"]), "split sets overlap"
    holdout = set(json.load(open(HOLDOUT_B_KEYS))["match_keys"])
    assert not ((fit | es | val) & holdout), "holdout B key entered the fit, the stop set or the validation set"
    print(f"fit matches {len(fit)}, early-stopping matches {len(es)}, validation matches {len(val)}; test and holdout B not selected", flush=True)

    keep = sorted(fit | es | val)
    G = (pl.scan_parquet(TRAIN_CACHE)
         .filter(pl.col("match_key").is_in(keep))
         .select(POS_KEY + ["label_ct_win"] + base + keep_cols)
         .with_columns([pl.col(c).cast(pl.Float32) for c in base])
         .join(pos, on=POS_KEY, how="inner")
         .collect())
    print(f"rows with positions: {G.height:,}", flush=True)
    Xtr, ytr = to_xy(G.filter(pl.col("match_key").is_in(sorted(fit))), feats)
    Xes, yes = to_xy(G.filter(pl.col("match_key").is_in(sorted(es))), feats)
    print(f"fit {Xtr.shape}, early stop {Xes.shape}", flush=True)
    m, best = W.fit_xgb(Xtr, ytr, Xes, yes, W.mono_for(feats), seed=a.seed, device=a.device)
    print(f"best iteration {best}", flush=True)
    del Xtr, Xes

    Wv = G.filter(pl.col("match_key").is_in(sorted(val)))
    Xv, yv = to_xy(Wv, feats)
    p = W.predict(m, best, Xv)
    col = "p"
    out = Wv.select(["match_key", "round"] + (["second"] if keep_cols else ["tick"]) + ["label_ct_win"]).with_columns(pl.Series(col, p))
    out.write_parquet(a.out)
    ll = float(-(yv * np.log(np.clip(p, 1e-15, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1))).mean())
    print(f"validation log loss {ll:.5f}; wrote {a.out} ({out.height:,} rows, {out['match_key'].n_unique()} matches)", flush=True)


if __name__ == "__main__":
    main()
