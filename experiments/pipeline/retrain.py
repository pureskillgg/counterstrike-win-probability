import json, os, gc, time
import numpy as np, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/g/pureskill/csds-cs2"
SCR = "/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
CACHE = os.environ.get("CACHE", ROOT + "/data/combined/cache_snapshot_2026.parquet")
TAG = os.environ.get("TAG", "retrain")
OUT_JSON = SCR + "/pipeline/" + TAG + "_result.json"
OUT_PRED = SCR + "/pipeline/" + TAG + "_test_preds.parquet"
meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
frozen = set(json.load(open(SCR + "/frozen_test_keys.json")))
KEYS = ["match_key", "round", "second"]

df = pl.read_parquet(CACHE, columns=list(dict.fromkeys(KEYS + ["label_ct_win"] + FEATS)))
uniq, inv = np.unique(df["match_key"].to_numpy(), return_inverse=True)
sp = np.array([("test" if u in frozen else man.get(u, "NA")) for u in uniq])
split = sp[inv]
del uniq, inv
print("rows", df.height, "| matches train", len(set(df["match_key"].filter(split == "train").to_list())) if False else "", flush=True)
counts = {s: int((split == s).sum()) for s in ["train", "val", "test"]}
print("row counts", counts, flush=True)
y = df["label_ct_win"].cast(pl.Float32).to_numpy()
X = df.select(FEATS).to_numpy().astype(np.float32)
keep_keys = df["match_key"].to_numpy()
del df
gc.collect()

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
dtr = xgb.QuantileDMatrix(X[split == "train"], label=y[split == "train"])
dva = xgb.QuantileDMatrix(X[split == "val"], label=y[split == "val"], ref=dtr)
params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
          "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": 0,
          "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
t = time.time()
bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
print("trained best_iteration", bst.best_iteration, "in", round(time.time() - t), "s", flush=True)
te = split == "test"
p = bst.predict(xgb.DMatrix(X[te]), iteration_range=(0, bst.best_iteration + 1))
yt = y[te].astype(np.float64)
res = {"tag": TAG, "cache": CACHE, "train_rows": counts["train"], "val_rows": counts["val"], "test_rows": counts["test"],
       "best_iteration": int(bst.best_iteration), "test_logloss": float(ll(yt, p).mean()),
       "test_auc": float(roc_auc_score(yt, p)), "test_ece10": float(ece(yt, p))}
print(json.dumps(res), flush=True)
json.dump(res, open(OUT_JSON, "w"), indent=1)
pl.DataFrame({"match_key": keep_keys[te], "label_ct_win": yt, "p_base": p}).write_parquet(OUT_PRED)
