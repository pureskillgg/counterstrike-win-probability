import json, ast, gc, time, os
import numpy as np, polars as pl, xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = "/mnt/g/pureskill/csds-cs2"
SCR = "/mnt/c/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
OUT = SCR + "/e"
CACHE = ROOT + "/data/combined/cache_snapshot_2026.parquet"
meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
MONO = ast.literal_eval(meta["monotone_constraints"]) if isinstance(meta["monotone_constraints"], str) else meta["monotone_constraints"]
man = json.load(open(ROOT + "/data/combined/split_manifest.json"))["assign"]
frozen = set(json.load(open(SCR + "/frozen_test_keys.json")))
KEYS = ["match_key", "round", "second"]
EQUIP = ["st_equip_ct", "st_equip_t", "st_equip_diff"]
NOEQ = [f for f in FEATS if f not in EQUIP]

df = pl.read_parquet(CACHE, columns=list(dict.fromkeys(KEYS + ["label_ct_win", "seconds_since_round_live", "bomb_planted", "alive_ct", "alive_t"] + FEATS)))
uniq, inv = np.unique(df["match_key"].to_numpy(), return_inverse=True)
sp = np.array([("test" if u in frozen else man.get(u, "NA")) for u in uniq])
split = sp[inv]
del uniq, inv
y = df["label_ct_win"].cast(pl.Float32).to_numpy()
Xf = df.select(FEATS).to_numpy().astype(np.float32)
meta_df = df.select(KEYS + ["label_ct_win", "seconds_since_round_live", "bomb_planted", "alive_ct", "alive_t"])
del df
gc.collect()
z = np.load(SCR + "/money/money_rows.npz")
money_ct = z["money_ct"]; money_t = z["money_t"]; money_diff = money_ct - money_t
MONEY = {"money_ct": money_ct, "money_t": money_t, "money_diff": money_diff}
print("rows", Xf.shape, "splits", {s: int((split == s).sum()) for s in ["train", "val", "test"]}, flush=True)

def ll(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))

def col_idx(names):
    return [FEATS.index(n) for n in names]

def build(mask, use_feats, extra):
    parts = [Xf[mask][:, col_idx(use_feats)]]
    if extra:
        parts.append(np.column_stack([MONEY[e][mask] for e in extra]))
    return np.hstack(parts)

MONEY_ALL = ["money_ct", "money_t", "money_diff"]
configs = [
    ("base59", FEATS, []),
    ("base59_money_all", FEATS, MONEY_ALL),
    ("base59_money_ct", FEATS, ["money_ct"]),
    ("base59_money_t", FEATS, ["money_t"]),
    ("base59_money_diff", FEATS, ["money_diff"]),
    ("noequip_all", NOEQ, []),
    ("noequip_ct_only", [f for f in FEATS if f != "st_equip_ct"], []),
    ("noequip_t_only", [f for f in FEATS if f != "st_equip_t"], []),
    ("noequip_diff_only", [f for f in FEATS if f != "st_equip_diff"], []),
    ("noequip_all_money", NOEQ, MONEY_ALL),
]
preds, results = {}, {}
for name, use_feats, extra in configs:
    t = time.time()
    mono = [int(MONO[f]) for f in use_feats] + [0] * len(extra)
    Xtr = build(split == "train", use_feats, extra); ytr = y[split == "train"]
    Xva = build(split == "val", use_feats, extra); yva = y[split == "val"]
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr)
    dva = xgb.QuantileDMatrix(Xva, label=yva, ref=dtr)
    del Xtr, Xva
    gc.collect()
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
              "max_depth": 4, "eta": 0.05, "min_child_weight": 20, "subsample": 0.8, "seed": 0,
              "monotone_constraints": "(" + ",".join(str(m) for m in mono) + ")"}
    bst = xgb.train(params, dtr, num_boost_round=3000, evals=[(dva, "val")], early_stopping_rounds=100, verbose_eval=False)
    te = split == "test"
    p = bst.predict(xgb.DMatrix(build(te, use_feats, extra)), iteration_range=(0, bst.best_iteration + 1))
    preds[name] = p
    yt = y[te].astype(np.float64)
    results[name] = {"best_iteration": int(bst.best_iteration), "test_logloss": float(ll(yt, p).mean()),
                     "test_auc": float(roc_auc_score(yt, p))}
    print(name, json.dumps(results[name]), round(time.time() - t), "s", flush=True)
    del dtr, dva, bst
    gc.collect()

te = split == "test"
out = meta_df.filter(pl.Series(te))
for k, v in preds.items():
    out = out.with_columns(pl.Series("p_" + k, v))
out = out.with_columns([pl.Series("money_ct", money_ct[te]), pl.Series("money_t", money_t[te])])
out.write_parquet(OUT + "/e1_preds.parquet")
json.dump(results, open(OUT + "/e1_results.json", "w"), indent=1)
print("saved", OUT + "/e1_preds.parquet", flush=True)
