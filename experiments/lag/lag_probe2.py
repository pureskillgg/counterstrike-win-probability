import json, ast, gc
import numpy as np, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score
ROOT = "/mnt/g/pureskill/csds-cs2"
OUT = "/mnt/c/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/lag"
meta = json.load(open(f"{ROOT}/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(f"{ROOT}/data/combined/split_manifest.json"))["assign"]
LAG_VARS = ["alive_ct", "alive_t", "st_hp_ct", "st_hp_t", "st_equip_ct", "st_equip_t", "st_armor_sum_ct", "st_armor_sum_t"]
LAGS = [1, 3, 5, 10]
KEYS = ["match_key", "round", "second"]
LAG_COLS = [f"{v}_lag{k}" for k in LAGS for v in LAG_VARS]
LAG_MONO = [1 if c.startswith("alive_ct") else (-1 if c.startswith("alive_t") else 0) for c in LAG_COLS]
BASE_MONO = [int(MONO[f]) for f in FEATS]
ALL_COLS = FEATS + LAG_COLS
def load(split):
    keys = [k for k, v in man.items() if v == split]
    df = (pl.scan_parquet(f"{ROOT}/data/combined/cache_snapshot_2026.parquet")
          .select(KEYS + ["label_ct_win"] + list(dict.fromkeys(FEATS + LAG_VARS)))
          .filter(pl.col("match_key").is_in(keys)).collect())
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in list(dict.fromkeys(FEATS + LAG_VARS))])
    lagv = df.select(KEYS + LAG_VARS)
    for k in LAGS:
        sh = lagv.with_columns((pl.col("second") + k).alias("second")).rename({v: f"{v}_lag{k}" for v in LAG_VARS})
        df = df.join(sh, on=KEYS, how="left")
    df = df.with_columns([pl.col(c).cast(pl.Float32) for c in LAG_COLS])
    X = df.select(ALL_COLS).to_numpy().astype(np.float32, copy=False)
    y = df["label_ct_win"].to_numpy().astype(np.float32)
    mk = df["match_key"].to_numpy()
    rnd = df["round"].to_numpy(); sec = df["second"].to_numpy()
    del df, lagv, sh; gc.collect()
    return X, y, mk, rnd, sec
def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e
print("loading train", flush=True); Xtr, ytr, _, _, _ = load("train")
print(f"train {Xtr.shape}", flush=True)
print("loading val", flush=True); Xva, yva, _, _, _ = load("val")
print("loading test", flush=True); Xte, yte, mkte, rndte, secte = load("test")
print(f"val {Xva.shape} test {Xte.shape}", flush=True)
def fit(cols_idx, mono):
    A = Xtr if cols_idx is None else np.ascontiguousarray(Xtr[:, cols_idx])
    B = Xva if cols_idx is None else np.ascontiguousarray(Xva[:, cols_idx])
    dtr = xgb.QuantileDMatrix(A, label=ytr); dva = xgb.QuantileDMatrix(B, label=yva, ref=dtr)
    del A, B
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": 0,
              "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
    import time; t = time.time()
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    print(f"  best_iteration {bst.best_iteration}, train time {time.time()-t:.0f}s", flush=True)
    del dtr, dva; gc.collect()
    return bst
def predict(bst, cols_idx):
    C = Xte if cols_idx is None else np.ascontiguousarray(Xte[:, cols_idx])
    return bst.predict(xgb.DMatrix(C), iteration_range=(0, bst.best_iteration + 1))
results, preds = {}, {}
base_idx = np.arange(len(FEATS))
print("fitting base59", flush=True); b0 = fit(base_idx, BASE_MONO); preds["base59"] = predict(b0, base_idx)
del b0; gc.collect()
print("fitting lag", flush=True); b1 = fit(None, BASE_MONO + LAG_MONO); preds["lag59plus32"] = predict(b1, None)
for name in ["base59", "lag59plus32"]:
    p = preds[name]
    results[name] = {"test_logloss": float(ll(yte, p).mean()), "test_auc": float(roc_auc_score(yte, p)), "test_ece10": float(ece(yte, p))}
    print(f"{name}: test LL {results[name]['test_logloss']:.5f}, AUC {results[name]['test_auc']:.4f}, ECE {results[name]['test_ece10']:.4f}", flush=True)
codes, inv = np.unique(mkte, return_inverse=True)
d = ll(yte, preds["lag59plus32"]) - ll(yte, preds["base59"])
sum_d = np.bincount(inv, weights=d); cnt = np.bincount(inv)
rng = np.random.default_rng(0); idx = rng.integers(0, len(codes), size=(2000, len(codes)))
boot = sum_d[idx].sum(1) / cnt[idx].sum(1)
results["paired_lag_minus_base"] = {"delta_logloss": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
import polars as pl2
pl2.DataFrame({"match_key": mkte, "round": rndte, "second": secte, "label_ct_win": yte,
               "p_base59": preds["base59"], "p_lag": preds["lag59plus32"]}).write_parquet(f"{OUT}/lag_test_preds.parquet")
json.dump(results, open(f"{OUT}/lag_results.json", "w"), indent=1)
print(json.dumps(results, indent=1), flush=True)
