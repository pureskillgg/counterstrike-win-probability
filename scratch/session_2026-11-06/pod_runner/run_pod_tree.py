"""RunPod pipeline check (pin 3f7fa640): tree models from the cache snapshot on the volume. Validation only.

Usage (on the pod, inside the pixi default env):
  python run_pod_tree.py --runs P59_s0 X83_EXPLORATORY_s0 [--pos cs2data/combined/cache_pos_2026_allmatch_EXPLORATORY.parquet]

Splits: train = manifest 'train' minus frozen-test keys minus the early-stop hash split; early stop = 10% of train matches by
crc32 hash; validation = the 1,168 validation keys. Test-labelled matches and frozen keys are never loaded into a matrix.
"""
import argparse
import json
import os
import time
import zlib

import numpy as np
import polars as pl
import xgboost as xgb

VOL = "/workspace/"
SNAP = VOL + "cs2data/combined/cache_snapshot_2026.parquet"
MAN = VOL + "cs2data/combined/split_manifest.json"
META = VOL + "cs2data/keys/matrix_meta.json"
VAL = VOL + "cs2data/keys/val_keys.parquet"
FROZEN = VOL + "cs2data/keys/frozen_test_keys.json"
OUT = VOL + "results/"
os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--pos", default=VOL + "cs2data/combined/cache_pos_2026_allmatch_EXPLORATORY.parquet")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

meta = json.load(open(META))
BASE, ALLF = meta["base"], meta["feats_all"]
assert len(BASE) == 59 and len(ALLF) == 83, (len(BASE), len(ALLF))
POSF = [f for f in ALLF if f not in BASE]
MONO = {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1}

man = json.load(open(MAN))["assign"]
frozen = set(json.load(open(FROZEN)))
val_keys = set(pl.read_parquet(VAL)["match_key"].to_list())
print("base", len(BASE), "pos", len(POSF), "manifest", len(man), "frozen", len(frozen), "val", len(val_keys), flush=True)


def split_of(k):
    if k in frozen:
        return None
    if k in val_keys:
        return "val"
    if man.get(k) == "train":
        return "es" if zlib.crc32(("es:" + k).encode()) % 100 < 10 else "train"
    return None


t0 = time.time()
df = pl.read_parquet(SNAP, columns=["match_key", "round", "tick", "second", "label_ct_win"] + BASE)
keys = df["match_key"].unique().to_list()
lk = {k: split_of(k) for k in keys}
df = df.with_columns(pl.col("match_key").replace_strict(lk, default=None).alias("split")).filter(pl.col("split").is_not_null())
print("snapshot rows kept", df.height, "in", round(time.time() - t0), "s", flush=True)

pos_cache = None
for run in a.runs:
    feats = BASE if run.startswith("P59") else ALLF
    d = df
    if feats is ALLF:
        if pos_cache is None:
            pos = pl.read_parquet(VOL + a.pos.split(VOL)[-1] if a.pos.startswith(VOL) else a.pos)
            pos_cache = pos.select(["match_key", "round", "tick"] + POSF)
        d = d.join(pos_cache, on=["match_key", "round", "tick"], how="inner")
        print(run, "rows with spatial features", d.height, flush=True)
    res = {"run": run, "seed": a.seed, "n_features": len(feats), "feature_source": "cache snapshot (base 59)" if feats is BASE else
           "cache snapshot + cache_pos_2026 ALL-MATCH centres (EXPLORATORY, not for results)"}
    parts = {}
    for s in ["train", "es", "val"]:
        sub = d.filter(pl.col("split") == s)
        parts[s] = (np.ascontiguousarray(sub.select(feats).to_numpy().astype(np.float32)),
                    sub["label_ct_win"].to_numpy().astype(np.float32))
        res[f"rows_{s}"] = int(sub.height)
        res[f"matches_{s}"] = int(sub["match_key"].n_unique())
    mono = tuple(MONO.get(f, 0) for f in feats)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 4, "eta": 0.03, "min_child_weight": 20, "subsample": 0.8, "seed": a.seed,
              "max_bin": 256, "monotone_constraints": mono}
    t1 = time.time()
    dtr = xgb.QuantileDMatrix(parts["train"][0], parts["train"][1], max_bin=256)
    des = xgb.QuantileDMatrix(parts["es"][0], parts["es"][1], ref=dtr)
    dva = xgb.QuantileDMatrix(parts["val"][0], parts["val"][1], ref=dtr)
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(des, "es")], early_stopping_rounds=100, verbose_eval=False)
    best = int(bst.best_iteration)
    p = bst.predict(dva, iteration_range=(0, best + 1))
    y = parts["val"][1].astype(np.float64)
    ll = float(-(y * np.log(np.clip(p, 1e-15, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))).mean())
    res.update({"best_iteration": best, "val_logloss": ll, "fit_seconds": round(time.time() - t1, 1),
                "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip(),
                "test_rows_loaded": 0})
    json.dump(res, open(OUT + f"{run}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["run", "n_features", "rows_train", "rows_es", "rows_val", "best_iteration", "val_logloss", "fit_seconds"]}), flush=True)
    del parts, dtr, des, dva
print("POD_RUNS_DONE", flush=True)
