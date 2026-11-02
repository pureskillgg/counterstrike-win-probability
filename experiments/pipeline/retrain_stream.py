"""Retrain the v4 feature set on the frozen split, streaming the cache so that large caches fit in memory.

Train rows: manifest train. Validation rows: manifest val. Test rows: the frozen test keys only, so every run is scored on the
same 3,059 matches. Matches in none of those groups (for example the test-assigned matches added after the freeze) are not used.

The cache is read in hash buckets. Each bucket's rows go to per-split memory-mapped arrays on disk, and XGBoost reads them through
a DataIter, so peak memory is one bucket plus the quantile sketch, not the whole cache.

Env: CACHE (parquet path), TAG (output prefix), NB (hash buckets, default 8), KEEP=1 to keep the work arrays on disk.
"""
import ast
import gc
import json
import os
import time
import zlib

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/g/pureskill/csds-cs2"
SCR = "/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
CACHE = os.environ.get("CACHE", ROOT + "/data/combined/cache_snapshot_2026.parquet")
TAG = os.environ.get("TAG", "stream")
NB = int(os.environ.get("NB", "8"))
KEEP = os.environ.get("KEEP", "0") == "1"
WORK = f"{ROOT}/data/retrain_work/{TAG}"
os.makedirs(WORK, exist_ok=True)
KEYS = ["match_key", "round", "second"]
SPLITS = {"train": 0, "val": 1, "test": 2}

meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
frozen = set(json.load(open(SCR + "/frozen_test_keys.json")))
F = len(FEATS)


TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "1"))  # learning curve: keep this fraction of the training matches (by a fixed hash)


def split_code(k):
    # frozen test first; manifest test matches that are not frozen are left out (-1)
    if k in frozen:
        return SPLITS["test"]
    code_ = {"train": SPLITS["train"], "val": SPLITS["val"]}.get(man.get(k), -1)
    if code_ == SPLITS["train"] and zlib.crc32(k.encode()) % 10000 >= TRAIN_FRAC * 10000:
        return -1
    return code_


# 1. the split of every match, and the row count per split
keys = pl.scan_parquet(CACHE).select("match_key").unique().collect()["match_key"].to_list()
lookup = pl.DataFrame({"match_key": keys, "split": [split_code(k) for k in keys]}).with_columns(pl.col("split").cast(pl.Int8))
counts = (pl.scan_parquet(CACHE).select("match_key").join(lookup.lazy(), on="match_key", how="inner")
          .group_by("split").len().collect())
n = {int(r["split"]): int(r["len"]) for r in counts.iter_rows(named=True)}
print("rows per split", {k: n.get(v, 0) for k, v in SPLITS.items()}, "left out", n.get(-1, 0), flush=True)

# 2. stream the cache into per-split memory-mapped arrays
mm = {}
for name, c in SPLITS.items():
    mm[name] = (np.lib.format.open_memmap(f"{WORK}/{name}_X.npy", mode="w+", dtype=np.float32, shape=(n[c], F)),
                np.lib.format.open_memmap(f"{WORK}/{name}_y.npy", mode="w+", dtype=np.float32, shape=(n[c],)))
offset = {name: 0 for name in SPLITS}
test_keys = []
for b in range(NB):
    chunk = (pl.scan_parquet(CACHE)
             .filter((pl.col("match_key").hash(seed=7) % NB) == b)
             .select(KEYS + ["label_ct_win"] + FEATS)
             .join(lookup.lazy(), on="match_key", how="inner")
             .collect())
    for name, c in SPLITS.items():
        sub = chunk.filter(pl.col("split") == c)
        if sub.height == 0:
            continue
        X, y = mm[name]
        s, e = offset[name], offset[name] + sub.height
        X[s:e] = sub.select(FEATS).with_columns([pl.col(f).cast(pl.Float32) for f in FEATS]).to_numpy()
        y[s:e] = sub["label_ct_win"].cast(pl.Float32).to_numpy()
        offset[name] = e
        if name == "test":
            test_keys.append(sub.select(KEYS + ["label_ct_win"]))
    print("bucket", b, "done", flush=True)
    del chunk
    gc.collect()
for name, c in SPLITS.items():
    assert offset[name] == n[c], (name, offset[name], n[c])


# 3. XGBoost reads the memory-mapped arrays in slices
class Stream(xgb.DataIter):
    def __init__(self, name, step=1_000_000):
        super().__init__()
        self.X, self.y = mm[name]
        self.step = step
        self.i = 0

    def next(self, input_data):
        s = self.i * self.step
        if s >= len(self.y):
            return 0
        e = min(s + self.step, len(self.y))
        input_data(data=np.asarray(self.X[s:e]), label=np.asarray(self.y[s:e]))
        self.i += 1
        return 1

    def reset(self):
        self.i = 0


def ll(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def ece(yy, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.sum() / len(p) * abs(p[m].mean() - yy[m].mean())
    return e


mono = [int(MONO[f]) for f in FEATS]
params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": int(os.environ.get("SEED", "0")),
          "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
t = time.time()
dtr = xgb.QuantileDMatrix(Stream("train"), max_bin=256)
dva = xgb.QuantileDMatrix(Stream("val"), ref=dtr)
bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
print("trained best_iteration", bst.best_iteration, "in", round(time.time() - t), "s", flush=True)
bst.save_model(f"{SCR}/pipeline/{TAG}_model.json")

# 4. score the frozen test rows, in the same order as they were written
te_X, te_y = mm["test"]
p = np.empty(len(te_y), dtype=np.float64)
for s in range(0, len(te_y), 1_000_000):
    e = min(s + 1_000_000, len(te_y))
    p[s:e] = bst.predict(xgb.DMatrix(np.asarray(te_X[s:e])), iteration_range=(0, bst.best_iteration + 1))
yt = te_y.astype(np.float64)
res = {"tag": TAG, "cache": CACHE, "train_rows": n[0], "val_rows": n[1], "test_rows": n[2],
       "best_iteration": int(bst.best_iteration), "test_logloss": float(ll(yt, p).mean()),
       "test_auc": float(roc_auc_score(yt, p)), "test_ece10": float(ece(yt, p))}
print(json.dumps(res), flush=True)
json.dump(res, open(f"{SCR}/pipeline/{TAG}_result.json", "w"), indent=1)
pl.concat(test_keys).with_columns(pl.Series("p_base", p)).write_parquet(f"{SCR}/pipeline/{TAG}_test_preds.parquet")
if not KEEP:
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)
