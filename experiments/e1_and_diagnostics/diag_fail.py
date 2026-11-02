import json, ast
import numpy as np, polars as pl, xgboost as xgb

ROOT = "G:/pureskill/csds-cs2"
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
meta = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
FEATS = ast.literal_eval(meta["features"]) if isinstance(meta["features"], str) else meta["features"]
best = int(meta["best_iteration"])
KEYS = ["match_key", "round", "second"]

p = pl.read_parquet(SCR + "/v4_test_preds_with_flags.parquet")

# (a) is the upset rate just the predicted rate? pre-plant 5v5 states, favorite framing
pre = p.filter((pl.col("planted") == 0) & (pl.col("ac") == 5) & (pl.col("at") == 5))
pre = pre.with_columns([
    pl.max_horizontal([pl.col("p"), 1 - pl.col("p")]).alias("fav_p"),
    (((pl.col("p") >= 0.5) & (pl.col("y") == 1)) | ((pl.col("p") < 0.5) & (pl.col("y") == 0))).cast(pl.Float64).alias("fav_won"),
])
bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
print("pre-plant 5v5, favorite framing:")
for lo, hi in bands:
    b = pre.filter((pl.col("fav_p") >= lo) & (pl.col("fav_p") < hi))
    if b.height == 0:
        continue
    print("  fav_p in [%.1f,%.1f): rows %d, predicted fav win %.3f, observed fav win %.3f, upset rate %.3f" % (
        lo, hi, b.height, b["fav_p"].mean(), b["fav_won"].mean(), 1 - b["fav_won"].mean()), flush=True)

# (b) which inputs the model leans on in its worst 1% of test states vs typical test states
lp = pl.read_parquet(SCR + "/lag/lag_test_preds.parquet").with_columns(pl.col("second").cast(pl.Int64))
lp = lp.with_columns(pl.col("p_base59").clip(1e-6, 1 - 1e-6).alias("pc")).with_columns(
    (-(pl.col("label_ct_win") * pl.col("pc").log() + (1 - pl.col("label_ct_win")) * (1 - pl.col("pc")).log())).alias("ll"))
worst = lp.sort("ll", descending=True).head(int(lp.height * 0.01))
wk = worst.select(KEYS)
cache = pl.read_parquet(ROOT + "/data/combined/cache_snapshot_2026.parquet", columns=KEYS + FEATS).with_columns(pl.col("second").cast(pl.Int64))
tk = lp.select(KEYS)
test_rows = cache.join(tk, on=KEYS, how="inner")
worst_rows = test_rows.join(wk, on=KEYS, how="inner")
sample = test_rows.sample(n=min(200000, test_rows.height), seed=0)
bst = xgb.Booster()
bst.load_model(ROOT + "/models/winprob_v4.json")
def contrib(df):
    X = df.select(FEATS).to_numpy().astype(np.float32)
    c = bst.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=(0, best + 1))
    return np.abs(c[:, :-1]).mean(0), c[:, :-1].mean(0)
w_abs, w_sgn = contrib(worst_rows)
t_abs, t_sgn = contrib(sample)
order = np.argsort(-(w_abs - t_abs))
print("\nworst 1%% states vs typical (mean |SHAP| worst, typical, signed worst):", worst_rows.height, "worst rows")
for i in order[:12]:
    print("  %-22s worst %.3f  typical %.3f  signed_worst %+.3f" % (FEATS[i], w_abs[i], t_abs[i], w_sgn[i]), flush=True)
