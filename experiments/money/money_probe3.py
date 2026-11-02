import json, ast, gc, glob, os, time
import numpy as np, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/g/pureskill/csds-cs2"
OUT = "/mnt/c/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/money"
ROWS = OUT + "/money_rows.npz"
CACHE = ROOT + "/data/combined/cache_snapshot_2026.parquet"
meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
KEYS = ["match_key", "round", "second"]

keys_df = (pl.read_parquet(CACHE, columns=KEYS + ["label_ct_win"]).with_row_index("ri")
           .with_columns(pl.col("second").cast(pl.Int64)))
N = keys_df.height
y_all = keys_df["label_ct_win"].cast(pl.Float32).to_numpy()
match_arr = keys_df["match_key"].to_numpy()

if not os.path.exists(ROWS):
    files = sorted(glob.glob(ROOT + "/data/status_1hz/*.parquet"))
    def read_status(f):
        s = pl.read_parquet(f, columns=["match_key", "round", "second", "side", "money", "health"])
        return s.with_columns(pl.col("second").cast(pl.Int64), pl.col("money").cast(pl.Float32), pl.col("health").cast(pl.Float32))
    cache_hp = pl.read_parquet(CACHE, columns=KEYS + ["st_hp_ct"]).with_columns(pl.col("second").cast(pl.Int64))
    s0 = read_status(files[0])
    best = None
    for sv in sorted(s0["side"].unique().to_list()):
        agg = s0.filter(pl.col("side") == sv).group_by(KEYS).agg(pl.col("health").sum().alias("hp_chk"))
        j = cache_hp.join(agg, on=KEYS, how="inner")
        if j.height == 0:
            continue
        err = float((j["hp_chk"] - j["st_hp_ct"]).abs().mean())
        print("side", sv, "matched", j.height, "mean abs diff vs st_hp_ct", round(err, 3), flush=True)
        if best is None or err < best[1]:
            best = (sv, err)
    CT_SIDE = best[0]
    T_SIDE = [s for s in sorted(s0["side"].unique().to_list()) if s != CT_SIDE][0]
    print("CT side", CT_SIDE, "T side", T_SIDE, flush=True)
    del s0, cache_hp
    gc.collect()
    money_ct = np.full(N, np.nan, dtype=np.float32)
    money_t = np.full(N, np.nan, dtype=np.float32)
    for i, f in enumerate(files):
        s = read_status(f)
        mk = s["match_key"].unique().to_list()
        sub = keys_df.filter(pl.col("match_key").is_in(mk))
        agg = s.filter(pl.col("health") > 0).group_by(KEYS + ["side"]).agg(pl.col("money").sum().alias("m"))
        ct = agg.filter(pl.col("side") == CT_SIDE).select(KEYS + [pl.col("m").alias("money_ct")])
        tt = agg.filter(pl.col("side") == T_SIDE).select(KEYS + [pl.col("m").alias("money_t")])
        j = sub.join(ct, on=KEYS, how="left").join(tt, on=KEYS, how="left")
        ri = j["ri"].to_numpy()
        money_ct[ri] = j["money_ct"].to_numpy().astype(np.float32)
        money_t[ri] = j["money_t"].to_numpy().astype(np.float32)
        del s, agg, ct, tt, sub, j
        gc.collect()
        if i % 10 == 0:
            print("aggregated", i + 1, "of", len(files), flush=True)
    np.savez(ROWS, money_ct=money_ct, money_t=money_t)
    print("saved money rows; coverage of all cache rows:", round(float(np.mean(~np.isnan(money_ct))), 4), flush=True)
    del money_ct, money_t
    gc.collect()

z = np.load(ROWS)
money_ct = z["money_ct"]
money_t = z["money_t"]
money_diff = money_ct - money_t

uniq, inv = np.unique(match_arr, return_inverse=True)
sp = np.array([man.get(u, "NA") for u in uniq])
split = sp[inv]
del match_arr, inv, uniq
gc.collect()
test_mask = split == "test"
cov = float(np.mean(~np.isnan(money_ct[test_mask])))
print("test-row money coverage:", round(cov, 4), flush=True)

Xf = pl.read_parquet(CACHE, columns=FEATS).to_numpy().astype(np.float32)
print("feature matrix", Xf.shape, flush=True)

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

def build(mask, with_money):
    if with_money:
        return np.column_stack([Xf[mask], money_ct[mask], money_t[mask], money_diff[mask]])
    return Xf[mask]

results, preds = {}, {}
for name, with_money in [("base59", False), ("base59_plus_money", True)]:
    t = time.time()
    mono = BASE_MONO + ([0, 0, 0] if with_money else [])
    Xtr = build(split == "train", with_money); ytr = y_all[split == "train"]
    Xva = build(split == "val", with_money); yva = y_all[split == "val"]
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr)
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
    del Xtr, Xva
    gc.collect()
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": 0,
              "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    Xte = build(test_mask, with_money); yte = y_all[test_mask]
    p = bst.predict(xgb.DMatrix(Xte), iteration_range=(0, bst.best_iteration + 1))
    preds[name] = p
    results[name] = {"best_iteration": int(bst.best_iteration), "test_logloss": float(ll(yte, p).mean()),
                     "test_auc": float(roc_auc_score(yte, p)), "test_ece10": float(ece(yte, p)), "test_rows": int(len(yte))}
    print(name, json.dumps(results[name]), "time", round(time.time() - t), "s", flush=True)
    del dtr, dva, Xte, bst
    gc.collect()

yte = y_all[test_mask].astype(np.float64)
d = ll(yte, preds["base59_plus_money"]) - ll(yte, preds["base59"])
mk = keys_df["match_key"].to_numpy()[test_mask]
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
