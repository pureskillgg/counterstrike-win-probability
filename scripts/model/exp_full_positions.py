"""Do player positions, view angles and blindness help? Full frozen split: every second, the same rows for both models.   (XGBoost, GPU when available)

  ...python.exe -I -X utf8 scripts\\model\\exp_full_positions.py [--store combined] [--device cuda]

base : the 59 snapshot inputs of the final model (data/<store>/cache_snapshot_2026.parquet)
pos  : base + the 24 position / view / blindness inputs (data/<store>/cache_pos_2026.parquet, built by make_pos_features.py --every 1)

Both are fitted with the same settings as wp_common.fit_xgb (depth 4, eta 0.05, min child weight 20, subsample 0.8, monotone alive counts,
early stopping after 100 rounds on the validation matches), on every second of the frozen training matches, and scored on every second of the
frozen test matches. The paired log-loss difference (pos minus base) carries a match-cluster interval, and the round-weighted difference is written next to it.
Memory: the snapshot and position tables are joined once into a file, one block of matches at a time. XGBoost reads the training and validation
data block by block into a QuantileDMatrix (about one byte per value), so no full float matrix of the training set is ever held in memory.
Rows where a position value is missing stay in: XGBoost handles nulls, and both models see the same rows.
Writes data/<store>/exp_full_positions.json.
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

sys.stdout.reconfigure(encoding="utf-8")
POS = ["c_site_hi", "c_site_lo", "t_site_hi", "t_site_lo", "c_nearest_site_mean", "t_nearest_site_mean", "c_spread", "t_spread", "ct_t_centroid_dist",
       "min_enemy_dist", "c_in_contact", "t_in_contact", "c_min_bomb_dist", "c_n_bomb_500", "t_min_bomb_dist", "defuse_slack", "c_sees_enemy",
       "t_sees_enemy", "c_exposed", "t_exposed", "c_blind_n", "t_blind_n", "c_blind_sum", "t_blind_sum"]
KEYS = ["match_key", "round", "tick"]
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", learning_rate=0.05, max_depth=4, min_child_weight=20, subsample=0.8,
              tree_method="hist", seed=0)


def joined_file(snap_path, pos_path, feats, out_path, chunk=1000):
    """snapshot + positions joined once, written to a file. The join runs one block of matches at a time, so only a block is in memory."""
    src = {"snap": [os.path.getsize(snap_path), os.path.getmtime(snap_path)], "pos": [os.path.getsize(pos_path), os.path.getmtime(pos_path)],
           "feats": feats, "POS": POS}
    meta = out_path + ".src.json"
    if os.path.exists(out_path) and os.path.exists(meta):
        with open(meta, encoding="utf-8") as f:
            if json.load(f) == src:
                return out_path
    if os.path.exists(out_path):
        print("joined table was built from other sources (or before the source record existed): rebuilding", flush=True)
    import pyarrow.parquet as pq
    snap = pl.scan_parquet(snap_path).select(KEYS + ["label_ct_win", "w_round"] + feats)
    pos = pl.scan_parquet(pos_path).select(KEYS + POS).with_columns([pl.col(c).cast(pl.Float32) for c in POS])
    mks = sorted(snap.select("match_key").unique().collect()["match_key"].to_list())
    writer = None
    for c0 in range(0, len(mks), chunk):
        ck = mks[c0:c0 + chunk]
        blk = snap.filter(pl.col("match_key").is_in(ck)).join(pos.filter(pl.col("match_key").is_in(ck)), on=KEYS, how="left").collect()
        tab = blk.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out_path + ".tmp", tab.schema, compression="zstd")
        writer.write_table(tab.cast(writer.schema))
        del blk, tab
        print(f"  joined {min(c0 + chunk, len(mks))} of {len(mks)} matches", flush=True)
    writer.close()
    os.replace(out_path + ".tmp", out_path)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump(src, f)
    return out_path


class Blocks(xgb.DataIter):
    """The rows of some matches, handed to XGBoost one block of matches at a time (never all at once)."""

    def __init__(self, lf, keys, cols, chunk=1500):
        super().__init__()
        self.lf, self.keys, self.cols, self.chunk = lf, keys, cols, chunk
        self.i = 0

    def next(self, input_data):
        if self.i * self.chunk >= len(self.keys):
            return 0
        ck = self.keys[self.i * self.chunk:(self.i + 1) * self.chunk]
        df = self.lf.filter(pl.col("match_key").is_in(ck)).select(["label_ct_win"] + [pl.col(c).cast(pl.Float32) for c in self.cols]).collect()
        input_data(data=df.select(self.cols).to_numpy().astype(np.float32, copy=False), label=df["label_ct_win"].to_numpy().astype(np.float32))
        self.i += 1
        del df
        return 1

    def reset(self):
        self.i = 0


def build(lf, keys, cols, chunk=1500):
    """Test rows as one float32 matrix with the match key and weight for scoring (the test set is read in full only after training)."""
    n = int(lf.filter(pl.col("match_key").is_in(keys)).select(pl.len()).collect().item())
    X = np.empty((n, len(cols)), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    w = np.empty(n, dtype=np.float64)
    mk = np.empty(n, dtype=object)
    i = 0
    for c0 in range(0, len(keys), chunk):
        ck = keys[c0:c0 + chunk]
        df = lf.filter(pl.col("match_key").is_in(ck)).select(["match_key", "label_ct_win", "w_round"] + [pl.col(c).cast(pl.Float32) for c in cols]).collect()
        m = df.height
        X[i:i + m] = df.select(cols).to_numpy()
        y[i:i + m] = df["label_ct_win"].to_numpy()
        w[i:i + m] = df["w_round"].to_numpy()
        mk[i:i + m] = df["match_key"].to_numpy()
        i += m
        del df
    return X, y, w, mk


def fit(lf, tr, va, cols, mono, device):
    dtr = xgb.QuantileDMatrix(Blocks(lf, tr, cols), max_bin=256)
    dva = xgb.QuantileDMatrix(Blocks(lf, va, cols), ref=dtr)
    t = time.time()
    bst = xgb.train(dict(PARAMS, device=device, monotone_constraints=tuple(mono)), dtr, num_boost_round=3000, evals=[(dva, "val")],
                    early_stopping_rounds=100, verbose_eval=False)
    return bst, int(bst.best_iteration), round(time.time() - t), int(dtr.num_row())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="combined")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", a.store)
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    cols83 = feats + POS
    joined = joined_file(os.path.join(d, "cache_snapshot_2026.parquet"), os.path.join(d, "cache_pos_2026.parquet"), feats,
                         os.path.join(d, "_joined_pos_2026.parquet"))
    lf = pl.scan_parquet(joined)
    cover = float(lf.select(pl.col("c_site_hi").is_not_null().mean()).collect().item())
    kdf = lf.select("match_key").unique().collect().with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"),
                                                                 pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    del kdf
    tr, va, te = sorted(sp["train"]), sorted(sp["val"]), sorted(sp["test"])
    mono = list(W.mono_for(feats))
    mono_pos = mono + [0] * len(POS)
    print(f"joined table ready; position coverage {cover:.4%}; test matches {len(te)} ({time.time() - t0:.0f}s)", flush=True)

    # 1) position model (83 inputs), streamed into XGBoost; then its test predictions
    bst_p, best_p, s_p, n_tr = fit(lf, tr, va, cols83, mono_pos, a.device)
    print(f"pos83 trained on {n_tr:,} seconds: best iteration {best_p} ({s_p}s)", flush=True)
    Xte, yte, wte, mkte = build(lf, te, cols83)
    te_sub = pl.DataFrame({"match_key": mkte.astype(str), "w_round": wte})
    p_pos = bst_p.predict(xgb.DMatrix(Xte), iteration_range=(0, best_p + 1))
    del Xte

    # 2) base model (59 inputs), same rows, same settings
    bst_b, best_b, s_b, n_tr_b = fit(lf, tr, va, feats, mono, a.device)
    print(f"base59 trained on {n_tr_b:,} seconds: best iteration {best_b} ({s_b}s)", flush=True)
    Xte59, _, _, _ = build(lf, te, feats)
    p_base = bst_b.predict(xgb.DMatrix(Xte59), iteration_range=(0, best_b + 1))
    del Xte59

    # 3) paired scores on the same test seconds
    rb = W.evaluate(te_sub, yte, p_base)
    rp = W.evaluate(te_sub, yte, p_pos, ref=p_base)
    out = {"test_matches": len(te), "train_rows": int(n_tr), "test_rows": int(len(yte)), "position_coverage": cover, "models": {}}
    out["models"]["base59"] = {"logloss": rb["logloss"], "logloss_ci95": rb["logloss_ci95"], "logloss_round_weighted": rb["logloss_round_weighted"],
                               "auc": rb["auc"], "ece10": rb["ece10"], "best_iteration": best_b, "fit_seconds": s_b}
    out["models"]["pos83"] = {"logloss": rp["logloss"], "logloss_ci95": rp["logloss_ci95"], "logloss_round_weighted": rp["logloss_round_weighted"],
                              "auc": rp["auc"], "ece10": rp["ece10"], "best_iteration": best_p, "fit_seconds": s_p,
                              "delta_vs_base59": rp["delta_vs_ref"], "delta_vs_base59_ci95": rp["delta_vs_ref_ci95"],
                              "delta_round_weighted_vs_base59": rp["logloss_round_weighted"] - rb["logloss_round_weighted"]}
    print(f"base59: log loss {rb['logloss']:.5f} {[round(x, 5) for x in rb['logloss_ci95']]} round-weighted {rb['logloss_round_weighted']:.5f}", flush=True)
    print(f"pos83: log loss {rp['logloss']:.5f} {[round(x, 5) for x in rp['logloss_ci95']]} round-weighted {rp['logloss_round_weighted']:.5f} "
          f"delta vs base {rp['delta_vs_ref']:+.5f} {[round(x, 5) for x in rp['delta_vs_ref_ci95']]}", flush=True)
    W.dump_json(out, os.path.join(d, "exp_full_positions.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
