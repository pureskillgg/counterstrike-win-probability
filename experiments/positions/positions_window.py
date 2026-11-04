"""Positions model predictions for the intern protocol window (memory-lean version).

Two feature sets, trained on the same frozen-split training rows as scripts/model/exp_positions.py:
  full : the 59 inputs of the final model + 24 position, view and blindness inputs (83 inputs)
  base : the 59 inputs only (the matched control)

Scored on the window's test-assigned matches (the rows in --window-rows), at every second the position table has.

Memory: only the needed columns are read, only the training and validation matches are loaded for fitting, and every feature is
cast to float32 before it becomes an array. An earlier version read the whole cache and ran out of memory.

    CS2_ROOT=G:/pureskill/csds-cs2 python -I -X utf8 positions_window.py --window-rows <rows.parquet> --out <preds.parquet> \
        [--features full|base] [--device cpu|cuda]

Writes match_key, round, second, label_ct_win and p_xgb83 (or p_base) for the scored seconds.
"""
import argparse
import os
import sys

ROOT = os.environ.get("CS2_ROOT", "/mnt/g/pureskill/csds-cs2")
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
WINDOW_CACHE = os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_oct23_31.parquet")


def to_xy(df, feats):
    X = df.select(feats).to_numpy()
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    y = df["label_ct_win"].to_numpy().astype(np.int64)
    return X, y


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos", default=os.path.join(ROOT, "data", "combined", "cache_pos_2026.parquet"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--features", choices=["full", "base"], default="full")
    a = ap.parse_args()

    # 1. which columns exist, and the feature list
    names = list(pl.read_parquet_schema(TRAIN_CACHE).keys())
    cache_base = [c for c in W.F6 if c in names] + [c for c in names if c.startswith("map_is_")] + [c for c in names if c.startswith("st_")]
    base = list(dict.fromkeys(cache_base))
    feats = base if a.features == "base" else base + GEO + VIEW + BLIND
    pos_feats = [c for c in feats if c not in names]  # the position-table inputs
    print(f"features {a.features}: {len(feats)} inputs ({len(pos_feats)} from the position table)", flush=True)

    pos = (pl.scan_parquet(a.pos).select(POS_KEY + pos_feats)
           .with_columns([pl.col(c).cast(pl.Float32) for c in pos_feats]))

    # 2. the frozen split, exactly as exp_positions.py does it: match-level, on the training cache's matches
    meta = pl.scan_parquet(TRAIN_CACHE).select(["match_key", "era", "match_date"]).unique(subset=["match_key"]).collect()
    sp = W.make_split("random", meta, seed=0)
    print(f"train matches {len(sp['train'])}, val {len(sp['val'])}", flush=True)

    # 3. training and validation rows only, joined to the position table, float32 throughout
    tv = sorted(sp["train"] | sp["val"])
    G = (pl.scan_parquet(TRAIN_CACHE)
         .filter(pl.col("match_key").is_in(tv))
         .select(POS_KEY + ["label_ct_win"] + base)
         .with_columns([pl.col(c).cast(pl.Float32) for c in base])
         .join(pos, on=POS_KEY, how="inner")
         .collect())
    print(f"training rows with positions {G.height:,}", flush=True)
    Xtr, ytr = to_xy(G.filter(pl.col("match_key").is_in(sorted(sp["train"]))), feats)
    Xva, yva = to_xy(G.filter(pl.col("match_key").is_in(sorted(sp["val"]))), feats)
    del G
    print(f"train {Xtr.shape}, val {Xva.shape}", flush=True)

    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device=a.device)
    print(f"best iteration {best}", flush=True)
    del Xtr, Xva
    m.get_booster().save_model(a.out.replace(".parquet", "_model.json"))

    # 4. the window: the expanded cache rows of the window's matches, joined to positions and the rows file
    rows = (pl.read_parquet(a.window_rows).select(["match", "round", "second"])
            .rename({"match": "match_key"}).with_columns(pl.col("second").cast(pl.Int64)))
    keys = rows["match_key"].unique().to_list()
    Wd = (pl.scan_parquet(WINDOW_CACHE)
          .filter(pl.col("match_key").is_in(keys))
          .select(POS_KEY + ["second", "label_ct_win"] + base)
          .with_columns([pl.col(c).cast(pl.Float32) for c in base] + [pl.col("second").cast(pl.Int64)])
          .join(pos, on=POS_KEY, how="inner")
          .join(rows.lazy(), on=["match_key", "round", "second"], how="inner")
          .collect())
    print(f"window rows with positions: {Wd.height:,} in {Wd['match_key'].n_unique()} matches", flush=True)
    Xw, yw = to_xy(Wd, feats)
    p = W.predict(m, best, Xw)
    col = "p_base" if a.features == "base" else "p_xgb83"
    out = Wd.select(["match_key", "round", "second", "label_ct_win"]).with_columns(pl.Series(col, p))
    out.write_parquet(a.out)
    print(f"window log loss {ll(yw.astype(np.float64), p):.5f}; wrote {a.out} ({out.height:,} rows)", flush=True)


if __name__ == "__main__":
    main()
