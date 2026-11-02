import json, ast, gc, glob, time
import numpy as np, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/g/pureskill/csds-cs2"
OUT = "/mnt/c/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/money"
meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
KEYS = ["match_key", "round", "second"]
MONEY_COLS = ["money_ct", "money_t", "money_diff"]

cache_cols = list(dict.fromkeys(KEYS + ["label_ct_win", "st_hp_ct", "st_hp_t"] + FEATS))
cache = pl.read_parquet(ROOT + "/data/combined/cache_snapshot_2026.parquet", columns=cache_cols)
cache = cache.with_columns(pl.col("second").cast(pl.Int64))
files = sorted(glob.glob(ROOT + "/data/status_1hz/*.parquet"))
print("status files", len(files), flush=True)

def read_status(f):
    s = pl.read_parquet(f, columns=["match_key", "round", "second", "side", "money", "health"])
    return s.with_columns(pl.col("second").cast(pl.Int64), pl.col("money").cast(pl.Float32), pl.col("health").cast(pl.Float32))

# identify the CT side value by agreement with the cache's team health sums (first file)
s0 = read_status(files[0])
sides = sorted(s0["side"].unique().to_list())
best = None
for sv in sides:
    agg = s0.filter(pl.col("side") == sv).group_by(KEYS).agg(pl.col("health").sum().alias("hp_chk"))
    j = cache.select(KEYS + ["st_hp_ct"]).join(agg, on=KEYS, how="inner")
    if j.height == 0:
        continue
    err = float((j["hp_chk"] - j["st_hp_ct"]).abs().mean())
    print("side", sv, "matched", j.height, "mean abs diff vs st_hp_ct", round(err, 3), flush=True)
    if best is None or err < best[1]:
        best = (sv, err)
CT_SIDE = best[0]
T_SIDE = [s for s in sides if s != CT_SIDE][0]
print("CT side", CT_SIDE, "T side", T_SIDE, flush=True)
del s0
gc.collect()

parts = []
for i, f in enumerate(files):
    s = read_status(f)
    agg = s.filter(pl.col("health") > 0).group_by(KEYS + ["side"]).agg(pl.col("money").sum().alias("m"))
    ct = agg.filter(pl.col("side") == CT_SIDE).select(KEYS + [pl.col("m").alias("money_ct")])
    tt = agg.filter(pl.col("side") == T_SIDE).select(KEYS + [pl.col("m").alias("money_t")])
    parts.append(ct.join(tt, on=KEYS, how="full", coalesce=True))
    del s, agg, ct, tt
    gc.collect()
    if i % 5 == 0:
        print("aggregated", i + 1, "of", len(files), flush=True)
team = pl.concat(parts)
del parts
gc.collect()
print("team money rows", team.height, flush=True)

aug = cache.join(team, on=KEYS, how="left").with_columns(
    (pl.col("money_ct") - pl.col("money_t")).alias("money_diff"))
del cache, team
gc.collect()
test_keys = {k for k, v in man.items() if v == "test"}
cov = float(aug.filter(pl.col("match_key").is_in(list(test_keys)))["money_ct"].is_not_null().mean())
print("test-row money coverage:", round(cov, 4), flush=True)

def split_arrays(name, cols):
    keys = [k for k, v in man.items() if v == name]
    d = aug.filter(pl.col("match_key").is_in(keys))
    X = d.select(cols).to_numpy().astype(np.float32)
    y = d["label_ct_win"].cast(pl.Float32).to_numpy()
    return X, y, d.select(["match_key"] + KEYS + ["label_ct_win"])

def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return e

BASE_MONO = [int(MONO[f]) for f in FEATS]
results, preds = {}, {}
tests_keys = None
for name, cols, mono in [("base59", FEATS, BASE_MONO), ("base59_plus_money", FEATS + MONEY_COLS, BASE_MONO + [0, 0, 0])]:
    t = time.time()
    Xtr, ytr, _ = split_arrays("train", cols)
    Xva, yva, _ = split_arrays("val", cols)
    Xte, yte, kte = split_arrays("test", cols)
    tests_keys = kte
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr)
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
    del Xtr, Xva
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": 0,
              "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    p = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1))
    preds[name] = p
    results[name] = {"best_iteration": int(bst.best_iteration), "test_logloss": float(ll(yte, p).mean()),
                     "test_auc": float(roc_auc_score(yte, p)), "test_ece10": float(ece(yte, p)), "test_rows": int(len(yte))}
    print(name, json.dumps(results[name]), "time", round(time.time() - t), "s", flush=True)
    del dtr, dva, Xte
    gc.collect()

yte = tests_keys["label_ct_win"].cast(pl.Float64).to_numpy()
d = ll(yte, preds["base59_plus_money"]) - ll(yte, preds["base59"])
mk = tests_keys["match_key"].to_numpy()
codes, inv = np.unique(mk, return_inverse=True)
sum_d = np.bincount(inv, weights=d)
cnt = np.bincount(inv)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(codes), size=(2000, len(codes)))
boot = sum_d[idx].sum(1) / cnt[idx].sum(1)
results["paired_money_minus_base"] = {"delta_logloss": float(d.mean()),
                                      "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                      "money_test_coverage": cov}
json.dump(results, open(OUT + "/money_results.json", "w"), indent=1)
print(json.dumps(results["paired_money_minus_base"]), flush=True)
