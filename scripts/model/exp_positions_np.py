"""Positions by round type on the GPU, with the joined table read once into numpy arrays.

Same design and outputs as exp_positions_groups.py (base59 and pos83 on every second of the training matches, early stopping on the
validation matches, every round group scored on the full test set, paired deltas with match-cluster intervals). The difference is memory:
the polars block reads of the first version grew to the WSL limit on the local machine, so here the joined table is read once with pyarrow
into preallocated float32 arrays, and XGBoost trains on the GPU from those arrays.

  ...python.exe -I -X utf8 scripts\\model\\exp_positions_np.py [--device cuda]

Writes data/combined/exp_positions_groups.json.
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
import pyarrow.parquet as pq
import xgboost as xgb

import wp_common as W
from exp_full_positions import POS, joined_file
from exp_pistol import BASE

sys.stdout.reconfigure(encoding="utf-8")
GROUPS = {"pistol": ([1, 13], None), "after_pistol": ([2, 14], None), "rest": (None, [1, 13, 2, 14])}


CHUNK = 1_000_000


class ArrIter(xgb.DataIter):
    """Feeds XGBoost the rows of an in-memory array in chunks, so no second full copy of the array is made."""

    def __init__(self, X, y, cols_n):
        super().__init__()
        self.X, self.y, self.cols_n = X, y, cols_n
        self.i = 0

    def next(self, input_data):
        lo = self.i * CHUNK
        if lo >= len(self.y):
            return 0
        hi = min(lo + CHUNK, len(self.y))
        input_data(data=np.ascontiguousarray(self.X[lo:hi, :self.cols_n]), label=self.y[lo:hi].astype(np.float32))
        self.i += 1
        return 1

    def reset(self):
        self.i = 0


def predict_chunked(bst, best, X, cols_n):
    """Predictions for the rows of X, one chunk at a time."""
    out = []
    for lo in range(0, len(X), CHUNK):
        out.append(bst.predict(xgb.DMatrix(np.ascontiguousarray(X[lo:lo + CHUNK, :cols_n])), iteration_range=(0, best + 1)))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()
    d = os.path.join(W.ROOT, "data", "combined")
    t0 = time.time()
    feats = json.load(open(os.path.join(W.ROOT, "models", "winprob_v3.meta.json"), encoding="utf-8"))["features"]
    cols83 = feats + POS
    joined = joined_file(os.path.join(d, "cache_snapshot_2026.parquet"), os.path.join(d, "cache_pos_2026.parquet"), feats,
                         os.path.join(d, "_joined_pos_2026.parquet"))

    # the manifest split, by match key (same rule as the other experiments)
    keys = pl.scan_parquet(joined).select("match_key").unique().collect()
    kdf = keys.with_columns(pl.lit(1.0).alias("w_round"), pl.lit(0).alias("label_ct_win"), pl.lit("2026").alias("era"), pl.lit("").alias("match_date"))
    sp = W.make_split("random", kdf, seed=0)
    code = {}
    for k in sp["train"]:
        code[k] = 0
    for k in sp["val"]:
        code[k] = 1
    for k in sp["test"]:
        code[k] = 2
    mid = {k: i for i, k in enumerate(sorted(code))}
    n_by = {0: 0, 1: 0, 2: 0}
    pf = pq.ParquetFile(joined)
    cols_read = ["match_key", "round", "label_ct_win", "w_round"] + cols83
    # pass 1: rows per split
    for batch in pf.iter_batches(batch_size=2_000_000, columns=["match_key"]):
        mk = batch.column("match_key").to_pylist()
        cs = np.fromiter((code[m] for m in mk), dtype=np.int8, count=len(mk))
        for s in (0, 1, 2):
            n_by[s] += int((cs == s).sum())
    print(f"rows: train {n_by[0]:,} validation {n_by[1]:,} test {n_by[2]:,} ({time.time() - t0:.0f}s)", flush=True)

    # pass 2: fill arrays
    # the feature arrays live in files on disk (memory-mapped): the operating system pages them in and out, so the run cannot run out of memory
    names = {0: "train", 1: "validation", 2: "test"}
    X = {s: np.lib.format.open_memmap(os.path.join(d, f"_posnp_X_{names[s]}.npy"), mode="w+", dtype=np.float32, shape=(n_by[s], len(cols83)))
         for s in (0, 1, 2)}
    Y = {s: np.empty(n_by[s], dtype=np.int64) for s in (0, 1, 2)}
    R = {s: np.empty(n_by[s], dtype=np.int64) for s in (0, 1, 2)}
    M = {s: np.empty(n_by[s], dtype=np.int64) for s in (0, 1, 2)}
    Wt = {s: np.empty(n_by[s], dtype=np.float64) for s in (0, 1, 2)}
    fill = {0: 0, 1: 0, 2: 0}
    for batch in pf.iter_batches(batch_size=2_000_000, columns=cols_read):
        mk = batch.column("match_key").to_pylist()
        cs = np.fromiter((code[m] for m in mk), dtype=np.int8, count=len(mk))
        mids = np.fromiter((mid[m] for m in mk), dtype=np.int64, count=len(mk))
        rounds = batch.column("round").to_numpy().astype(np.int64)
        ys = batch.column("label_ct_win").to_numpy().astype(np.int64)
        ws = batch.column("w_round").to_numpy().astype(np.float64)
        block = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in cols83])  # nulls become NaN (XGBoost: missing)
        for s in (0, 1, 2):
            m = cs == s
            c = int(m.sum())
            if c == 0:
                continue
            i = fill[s]
            X[s][i:i + c] = block[m]
            Y[s][i:i + c] = ys[m]
            R[s][i:i + c] = rounds[m]
            M[s][i:i + c] = mids[m]
            Wt[s][i:i + c] = ws[m]
            fill[s] += c
        del block, mk, cs, mids, rounds, ys, ws
    print(f"arrays filled ({time.time() - t0:.0f}s)", flush=True)

    mono = list(W.mono_for(feats))
    mono_pos = mono + [0] * len(POS)

    def train(cols_n, mono_c, name):
        dtr = xgb.QuantileDMatrix(ArrIter(X[0], Y[0], cols_n), max_bin=256)
        dva = xgb.QuantileDMatrix(ArrIter(X[1], Y[1], cols_n), ref=dtr)
        t = time.time()
        bst = xgb.train(dict(BASE, seed=0, device=a.device, monotone_constraints=tuple(mono_c)), dtr, num_boost_round=3000,
                        evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
        best = int(bst.best_iteration)
        print(f"{name} trained: best {best} ({round(time.time() - t)}s, {time.time() - t0:.0f}s)", flush=True)
        return bst, best

    # pos83 first (uses the full arrays), then base59 (its columns are the first 59 of the same arrays)
    bst_p, best_p = train(len(cols83), mono_pos, "pos83")
    pp = {s: predict_chunked(bst_p, best_p, X[s], len(cols83)) for s in (1, 2)}
    del bst_p
    bst_b, best_b = train(len(feats), mono, "base59")
    pb = {s: predict_chunked(bst_b, best_b, X[s], len(feats)) for s in (1, 2)}
    del bst_b
    print(f"predictions done ({time.time() - t0:.0f}s)", flush=True)

    def mask_for(r, spec):
        inc, exc = spec
        if inc is not None:
            return np.isin(r, inc)
        return ~np.isin(r, exc)

    out = {"base_best_iteration": best_b, "pos_best_iteration": best_p, "device": a.device, "groups": {}}
    for s, split_name in ((1, "validation"), (2, "test")):
        sub = pl.DataFrame({"match_key": M[s], "w_round": Wt[s]})
        y, r, p_b, p_p = Y[s], R[s], pb[s], pp[s]
        out["groups"][split_name] = {}
        ov = W.evaluate(sub, y, p_p, ref=p_b)
        out["groups"][split_name]["overall"] = {"seconds": int(len(y)), "base59": W.evaluate(sub, y, p_b)["logloss"], "pos83": ov["logloss"],
                                                 "delta": ov["delta_vs_ref"], "delta_ci95": ov["delta_vs_ref_ci95"]}
        for g, spec in GROUPS.items():
            m = mask_for(r, spec)
            sg = sub.filter(pl.Series(m))
            rg = W.evaluate(sg, y[m], p_p[m], ref=p_b[m])
            out["groups"][split_name][g] = {"seconds": int(m.sum()), "base59": W.evaluate(sg, y[m], p_b[m])["logloss"], "pos83": rg["logloss"],
                                            "delta": rg["delta_vs_ref"], "delta_ci95": rg["delta_vs_ref_ci95"]}
        print(f"{split_name}:", flush=True)
        for g in ("overall", *GROUPS):
            e = out["groups"][split_name][g]
            print(f"  {g:13s} base59 {e['base59']:.5f}  pos83 {e['pos83']:.5f}  delta {e['delta']:+.5f} [{e['delta_ci95'][0]:+.5f}, {e['delta_ci95'][1]:+.5f}]",
                  flush=True)
    W.dump_json(out, os.path.join(d, "exp_positions_groups.json"))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
