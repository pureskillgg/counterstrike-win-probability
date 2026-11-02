import json, ast, os
import numpy as np, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score
ROOT = "/mnt/g/pureskill/csds-cs2"
OUT = "/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/lag"
meta = json.load(open(f"{ROOT}/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(f"{ROOT}/data/combined/split_manifest.json"))["assign"]
LAG_VARS = ["alive_ct", "alive_t", "st_hp_ct", "st_hp_t", "st_equip_ct", "st_equip_t", "st_armor_sum_ct", "st_armor_sum_t"]
LAGS = [1, 3, 5, 10]
KEYS = ["match_key", "round", "second"]
base = pl.read_parquet(f"{ROOT}/data/combined/cache_snapshot_2026.parquet", columns=KEYS + ["label_ct_win"] + FEATS)
assert base.select(KEYS).is_duplicated().sum() == 0, "duplicate (match, round, second) keys"
aug = base
for k in LAGS:
    sh = base.select(KEYS + LAG_VARS).with_columns((pl.col("second") + k).alias("second"))
    sh = sh.rename({v: f"{v}_lag{k}" for v in LAG_VARS})
    aug = aug.join(sh, on=KEYS, how="left")
LAG_COLS = [f"{v}_lag{k}" for k in LAGS for v in LAG_VARS]
LAG_MONO = [1 if c.startswith("alive_ct") else (-1 if c.startswith("alive_t") else 0) for c in LAG_COLS]
def part(name):
    keys = [k for k, v in man.items() if v == name]
    return aug.filter(pl.col("match_key").is_in(keys))
tr, va, te = part("train"), part("val"), part("test")
print(f"train {tr.height} rows / {tr['match_key'].n_unique()} matches; val {va.height}; test {te.height} / {te['match_key'].n_unique()}", flush=True)
def xy(df, cols):
    return df.select(cols).to_numpy().astype(np.float32), df["label_ct_win"].cast(pl.Float64).to_numpy()
def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e
def fit(cols, mono):
    Xtr, ytr = xy(tr, cols); Xva, yva = xy(va, cols)
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr); dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
    del Xtr, Xva
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": 0,
              "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
    t = __import__("time").time()
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    print(f"  trained {len(cols)} features, best_iteration {bst.best_iteration}, {__import__('time').time()-t:.0f}s", flush=True)
    return bst
results, preds = {}, {}
for name, cols, mono in [("base59", FEATS, [int(m) for m in [MONO[f] for f in FEATS]]),
                         ("lag59plus32", FEATS + LAG_COLS, [int(m) for m in [MONO[f] for f in FEATS]] + LAG_MONO)]:
    print(f"fitting {name}", flush=True)
    bst = fit(cols, mono)
    Xte, yte = xy(te, cols)
    p = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1))
    results[name] = {"test_logloss": float(ll(yte, p).mean()), "test_auc": float(roc_auc_score(yte, p)),
                     "test_ece10": float(ece(yte, p)), "best_iteration": int(bst.best_iteration)}
    preds[name] = p
    print(f"  {name}: test LL {results[name]['test_logloss']:.5f}, AUC {results[name]['test_auc']:.4f}", flush=True)
    del Xte
keys_df = te.select(KEYS + ["label_ct_win"])
y = keys_df["label_ct_win"].cast(pl.Float64).to_numpy()
d = ll(y, preds["lag59plus32"]) - ll(y, preds["base59"])
mk = keys_df["match_key"].to_numpy()
codes, inv = np.unique(mk, return_inverse=True)
sum_d = np.bincount(inv, weights=d); cnt = np.bincount(inv)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(codes), size=(2000, len(codes)))
boot = sum_d[idx].sum(1) / cnt[idx].sum(1)
results["paired_lag_minus_base"] = {"delta_logloss": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
keys_df.with_columns([pl.Series("p_base59", preds["base59"]), pl.Series("p_lag", preds["lag59plus32"])]).write_parquet(f"{OUT}/lag_test_preds.parquet")
json.dump(results, open(f"{OUT}/lag_results.json", "w"), indent=1)
print(json.dumps(results, indent=1), flush=True)
