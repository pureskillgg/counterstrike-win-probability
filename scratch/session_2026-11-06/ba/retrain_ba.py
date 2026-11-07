"""Before/after of the Oct 21-25 batch, validation only (pin: decision 5 Nov 2026 night).

Same recipe as experiments/pipeline/retrain_stream.py (winprob_v4 features, XGBoost, streamed cache), with three changes:
  - the frozen test keys are excluded entirely (no test rows are written, loaded or scored);
  - BATCH_MODE=none : the batch matches are excluded from train and validation (arm A);
    BATCH_MODE=train: the batch train matches are added to train; the batch validation matches stay excluded (arm B);
  - the validation set is the manifest validation matches minus the batch, identical in both arms.
Env: BATCH_MODE, TAG, NB, SEED. Writes only under the scratch folder.
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

ROOT = "G:/pureskill/csds-cs2"
SCR = "C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/ba"
CACHE = ROOT + "/data/combined/cache_snapshot_2026_oct23_31.parquet"
TAG = os.environ["TAG"]
BATCH_MODE = os.environ["BATCH_MODE"]
assert BATCH_MODE in ("none", "train")
NB = int(os.environ.get("NB", "16"))
SEED = int(os.environ.get("SEED", "0"))
WORK = f"{SCR}/work_{TAG}"
os.makedirs(WORK, exist_ok=True)
KEYS = ["match_key", "round", "second"]
SPLITS = {"train": 0, "val": 1}

meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
frozen = set(json.load(open(ROOT + "/data/benchmark/frozen_test_keys.json")))
batch = set(l.strip() for l in open(SCR + "/batch_keys.txt") if l.strip())
F = len(FEATS)
print("features", F, "batch keys", len(batch), "frozen keys", len(frozen), flush=True)


def split_code(k):
    if k in frozen:
        return -1
    if k in batch:
        return 0 if (BATCH_MODE == "train" and man.get(k) == "train") else -1
    return {"train": 0, "val": 1}.get(man.get(k), -1)


keys = pl.scan_parquet(CACHE).select("match_key").unique().collect()["match_key"].to_list()
lookup = pl.DataFrame({"match_key": keys, "split": [split_code(k) for k in keys]}).with_columns(pl.col("split").cast(pl.Int8))
counts = (pl.scan_parquet(CACHE).select("match_key").join(lookup.lazy(), on="match_key", how="inner")
          .group_by("split").len().collect())
n = {int(r["split"]): int(r["len"]) for r in counts.iter_rows(named=True)}
m_count = lookup.group_by("split").len()
mc = {int(r["split"]): int(r["len"]) for r in m_count.iter_rows(named=True)}
print("rows per split", {k: n.get(v, 0) for k, v in SPLITS.items()}, "matches", {k: mc.get(v, 0) for k, v in SPLITS.items()}, flush=True)

mm = {}
for name, c in SPLITS.items():
    mm[name] = (np.lib.format.open_memmap(f"{WORK}/{name}_X.npy", mode="w+", dtype=np.float32, shape=(n[c], F)),
                np.lib.format.open_memmap(f"{WORK}/{name}_y.npy", mode="w+", dtype=np.float32, shape=(n[c],)))
offset = {name: 0 for name in SPLITS}
for b in range(NB):
    chunk = (pl.scan_parquet(CACHE)
             .filter((pl.col("match_key").hash(seed=7) % NB) == b)
             .select(KEYS + ["label_ct_win"] + FEATS)
             .join(lookup.lazy(), on="match_key", how="inner")
             .filter(pl.col("split") >= 0)
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
    print("bucket", b, "done", flush=True)
    del chunk
    gc.collect()
for name, c in SPLITS.items():
    assert offset[name] == n[c], (name, offset[name], n[c])


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


mono = [int(MONO[f]) for f in FEATS]
params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": SEED,
          "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
t = time.time()
dtr = xgb.QuantileDMatrix(Stream("train"), max_bin=256)
dva = xgb.QuantileDMatrix(Stream("val"), ref=dtr)
bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
res = {"tag": TAG, "batch_mode": BATCH_MODE, "seed": SEED, "train_rows": n[0], "val_rows": n[1],
       "train_matches": mc.get(0, 0), "val_matches": mc.get(1, 0),
       "best_iteration": int(bst.best_iteration), "val_logloss_at_best": float(bst.best_score),
       "fit_seconds": round(time.time() - t)}
print(json.dumps(res), flush=True)
json.dump(res, open(f"{SCR}/result_{TAG}.json", "w"), indent=1)
import shutil
shutil.rmtree(WORK, ignore_errors=True)
