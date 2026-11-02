"""Positions model (the 59 inputs + 24 position, view and blindness inputs) predictions for the intern protocol window.

Trained on the frozen split's training matches, exactly as scripts/model/exp_positions.py does, so it is comparable with v4 and v6.
Scored on the window's test-assigned matches (the rows in --window-rows) at the seconds where the position table exists (every other second).

    CS2_ROOT=/mnt/g/pureskill/csds-cs2 ~/venvs/csds/bin/python -I -X utf8 positions_window.py \
        --window-rows <rows_intern.parquet> --out <preds.parquet>

Writes match_key, round, second, label_ct_win and p_xgb83 for the scored seconds.
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


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos", default=os.path.join(ROOT, "data", "combined", "cache_pos_2026.parquet"))
    a = ap.parse_args()

    pos = pl.read_parquet(a.pos)
    pos_cols = [c for c in pos.columns if c not in POS_KEY]

    # 1. training: the frozen cache, joined to positions, split exactly as exp_positions.py does it
    G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026.parquet"))
    G = G.join(pos, on=POS_KEY, how="inner")
    sp = W.make_split("random", G, seed=0)
    ST = [c for c in G.columns if c.startswith("st_")]
    base = W.F6 + [c for c in G.columns if c.startswith("map_is_")] + ST
    feats = base + GEO + VIEW + BLIND
    print(f"training positions rows {G.height:,}; train matches {len(sp['train'])}, val {len(sp['val'])}; inputs {len(feats)}", flush=True)
    _, Xtr, ytr = W.arrays(G, sp["train"], feats, stride=1)
    _, Xva, yva = W.arrays(G, sp["val"], feats, stride=1)
    del G
    m, best = W.fit_xgb(Xtr, ytr, Xva, yva, W.mono_for(feats), seed=0, device="cuda")
    print(f"best iteration {best}", flush=True)
    m.get_booster().save_model(a.out.replace(".parquet", "_model.json"))

    # 2. the window: the expanded cache rows of the window's matches, joined to positions
    rows = pl.read_parquet(a.window_rows).select(["match", "round", "second"]).rename({"match": "match_key"})
    keys = rows["match_key"].unique().to_list()
    W_cache = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_oct23_31.parquet"))
               .filter(pl.col("match_key").is_in(keys)).collect())
    W_cache = W_cache.join(pos, on=POS_KEY, how="inner").with_columns(pl.col("second").cast(pl.Int64))
    W_cache = W_cache.join(rows.with_columns(pl.col("second").cast(pl.Int64)), on=["match_key", "round", "second"], how="inner")
    print(f"window rows with positions: {W_cache.height:,} in {W_cache['match_key'].n_unique()} matches", flush=True)
    sub, X, y = W.arrays(W_cache, set(W_cache["match_key"].to_list()), feats, stride=1)
    p = W.predict(m, best, X)
    out = sub.select(["match_key", "round", "second", "label_ct_win"]).with_columns(pl.Series("p_xgb83", p))
    out.write_parquet(a.out)
    print(f"window log loss {ll(y.astype(np.float64), p):.5f}; wrote {a.out} ({out.height:,} rows)", flush=True)


if __name__ == "__main__":
    main()
