"""Tabular reference models on EXACTLY the rows of the graph model (data/gnn/index.parquet), so the comparison is paired.   (CUDA XGBoost)

  ...tabicl\\python.exe -I -X utf8 scripts\\model\\gnn\\xgb_reference.py

  xgb59  the final model's 59 inputs, same settings as winprob_v3 (depth 4, eta 0.05, min child weight 20, early stopping on validation matches)
  xgb83  the 59 inputs + the 24 position / view / blindness inputs of exp_positions.py
Trained on all frozen-split training rows of the index (every other second), scored on the test rows. Writes data/gnn/pred_xgb59.npy, pred_xgb83.npy (+ _rows.npy) and result_xgb*.json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import polars as pl

import wp_common as W

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
POS = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist", "min_enemy_dist",
       "c_in_contact", "t_in_contact", "c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist", "defuse_slack", "c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed",
       "c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]


def main():
    g = os.path.join(ROOT, "data", "gnn")
    t0 = time.time()
    meta = json.load(open(os.path.join(g, "meta.json"), encoding="utf-8"))
    idx = pl.read_parquet(os.path.join(g, "index.parquet"))
    sp = W.make_split("random", idx.select("match_key").unique().with_columns(pl.lit(1).alias("w_round"), pl.lit(0).alias("label_ct_win"), pl.lit("2026").alias("era"), pl.lit("").alias("match_date")), seed=0)
    code = {k: 0 for k in sp["train"]} | {k: 1 for k in sp["val"]} | {k: 2 for k in sp["test"]}
    split = np.array([code.get(k, -1) for k in idx["match_key"].to_list()], dtype=np.int8)
    y = idx["label_ct_win"].to_numpy().astype(np.int64)
    G59 = np.load(os.path.join(g, "G59.npy"), mmap_mode="r")
    feats59 = meta["G59_names"]
    pos = pl.read_parquet(os.path.join(ROOT, "data", "combined", "cache_pos_2026.parquet"), columns=["match_key", "round", "tick"] + POS)
    PX = idx.select("match_key", "round", "tick").join(pos, on=["match_key", "round", "tick"], how="left").select(POS).to_numpy().astype(np.float32)
    del pos
    tr, va, te = (np.where(split == s)[0] for s in (0, 1, 2))
    print(f"rows {len(idx):,}: train {len(tr):,} val {len(va):,} test {len(te):,} ({time.time() - t0:.0f}s)", flush=True)
    sub = idx[te].select("match_key", "round", "tick", "w_round")
    out = {}
    for name, with_pos in (("xgb59", False), ("xgb83", True)):
        t1 = time.time()
        feats = feats59 + (POS if with_pos else [])
        def mat(rows):
            A = np.asarray(G59[rows], dtype=np.float32)
            return np.hstack([A, PX[rows]]) if with_pos else A
        Xtr, Xva = mat(tr), mat(va)
        m, best = W.fit_xgb(Xtr, y[tr], Xva, y[va], W.mono_for(feats), seed=0, device="cuda")
        del Xtr, Xva
        p = W.predict(m, best, mat(te))
        np.save(os.path.join(g, f"pred_{name}.npy"), p.astype(np.float64))
        np.save(os.path.join(g, f"pred_{name}_rows.npy"), te)
        r = W.evaluate(sub, y[te], p.astype(np.float64))
        out[name] = {"inputs": len(feats), "best_iteration": int(best), "test": {k: r[k] for k in ("n_matches", "n_rows", "logloss", "logloss_ci95", "auc", "ece10", "ece10_ci95")}, "fit_seconds": round(time.time() - t1)}
        json.dump(out[name], open(os.path.join(g, f"result_{name}.json"), "w", encoding="utf-8"), indent=1)
        print(f"{name}: {len(feats)} inputs, best iteration {best}: test log loss {r['logloss']:.4f} {[round(x, 4) for x in r['logloss_ci95']]} AUC {r['auc']:.4f} ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
