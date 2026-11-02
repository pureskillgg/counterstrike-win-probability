"""Build intern-grader rows (README 'The rows') for the October 14-25 upload days.

Status: it runs, but on the frozen cache it returns 0 rows, because that cache has no matches after October 13.
It needs the expanded cache, and the test-assigned October 14-25 matches in it, before the grade means anything.

Rows are the every-other-second test rows that the GNN and xgb83 predictions cover, so every model is scored on the same seconds.
v4 and v6 predictions are recomputed from the frozen cache; GRU and lag predictions come from their saved test files.
"""
import ast
import json
import os

import numpy as np
import polars as pl
import xgboost as xgb

ROOT = "G:/pureskill/csds-cs2"
SCR = "C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
G = ROOT + "/data/gnn"
OUT = SCR + "/gdt"
os.makedirs(OUT, exist_ok=True)
MAPS = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke"]
D0, D1 = "2026/10/14", "2026/10/25"
KEY = ["match_key", "round", "second"]


def feats_of(meta):
    f = meta["features"]
    return ast.literal_eval(f) if isinstance(f, str) else list(f)


m4 = json.load(open(ROOT + "/models/winprob_v4.meta.json"))
m6 = json.load(open(ROOT + "/models/winprob_v6_drop_heavy.meta.json"))
F4, F6 = feats_of(m4), feats_of(m6)
print("v6 test block:", str(m6.get("test"))[:300], flush=True)

# 1. the every-other-second test rows that the GNN and xgb83 predictions cover
idx = pl.read_parquet(G + "/index.parquet", columns=["rid", "match_key", "round", "second", "label_ct_win", "map_name", "upload_day"])
rows = np.load(G + "/pred_xgb83_rows.npy")
for n in ["pred_gnn_hyb1", "pred_gnn_set"]:
    assert np.array_equal(rows, np.load(G + f"/{n}_rows.npy")), n
preds = pl.DataFrame({"rid": rows.astype(np.uint32)})
for n, col in [("pred_xgb83", "p_xgb83"), ("pred_gnn_hyb1", "p_gnn_hyb1"), ("pred_gnn_set", "p_gnn_set")]:
    preds = preds.with_columns(pl.Series(col, np.load(f"{G}/{n}.npy").astype(np.float64)))
base = idx.join(preds, on="rid", how="inner")
print("covered test rows:", base.height, flush=True)
base = base.filter((pl.col("upload_day") >= D0) & (pl.col("upload_day") <= D1) & pl.col("map_name").is_in(MAPS))
base = base.with_columns(pl.col("second").cast(pl.Int64))
print("Oct 14-25, seven maps:", base.height, "rows,", base["match_key"].n_unique(), "matches", flush=True)

# 2. the cache rows for those matches: features, equipment, bomb
keys = base["match_key"].unique().to_list()
need = sorted(set(F4) | set(F6) | {"match_key", "round", "second", "alive_ct", "alive_t", "st_equip_ct", "st_equip_t",
                                   "bomb_planted", "seconds_since_plant"})
cache = (pl.scan_parquet(ROOT + "/data/combined/cache_snapshot_2026.parquet")
         .filter(pl.col("match_key").is_in(keys)).select(need).with_columns(pl.col("second").cast(pl.Int64)))
starts = (cache.filter(pl.col("second") == 0).select(["match_key", "round", "alive_ct", "alive_t"]).collect()
          .filter((pl.col("alive_ct") == 5) & (pl.col("alive_t") == 5)).select(["match_key", "round"]))
print("5v5 round starts:", starts.height, flush=True)
cache_df = cache.join(base.lazy().select(KEY), on=KEY, how="inner").collect()
df = base.join(cache_df, on=KEY, how="inner").join(starts, on=["match_key", "round"], how="inner")
print("rows after cache join and 5v5 start:", df.height, flush=True)

# 3. the match filter the grader asks for (in-scope matches, max_rounds 24) where the matches table has those columns
mt = pl.read_parquet(ROOT + "/data/combined/matches.parquet")
print("matches columns:", mt.columns, flush=True)
if "match_key" in mt.columns:
    keep = mt.select("match_key")
    if "in_scope" in mt.columns:
        keep = mt.filter(pl.col("in_scope") == True).select("match_key")  # noqa: E712
    if "max_rounds_final" in mt.columns:
        keep = keep.join(mt.filter(pl.col("max_rounds_final") == 24).select("match_key"), on="match_key", how="inner")
    df = df.join(keep, on="match_key", how="inner")
    print("after matches filter:", df["match_key"].n_unique(), "matches", flush=True)

# 4. predictions: v4 and v6 recomputed on these rows; GRU and lag from their saved test files
def predict(model_path, meta, feats, frame):
    bst = xgb.Booster()
    bst.load_model(model_path)
    X = frame.select(feats).to_numpy().astype(np.float32)
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, int(meta["best_iteration"]) + 1))


df = df.with_columns(pl.Series("p_v4", predict(ROOT + "/models/winprob_v4.json", m4, F4, df)),
                     pl.Series("p_v6", predict(ROOT + "/models/winprob_v6_drop_heavy.json", m6, F6, df)))
lag = pl.read_parquet(SCR + "/lag/lag_test_preds.parquet", columns=KEY + ["p_lag"]).with_columns(pl.col("second").cast(pl.Int64))
gru = pl.read_parquet(SCR + "/gru/gru_full_test_preds.parquet", columns=KEY + ["p_gru"]).with_columns(pl.col("second").cast(pl.Int64))
df = df.join(lag, on=KEY, how="left").join(gru, on=KEY, how="left")
for c in ["p_lag", "p_gru"]:
    print(c, "missing rows:", df.filter(pl.col(c).is_null()).height, flush=True)

# 5. the grader's columns; since_plant is 0 before the plant (the README says it is ignored there)
out = df.drop_nulls(["p_v4", "p_v6", "p_xgb83", "p_gnn_hyb1", "p_gnn_set", "p_gru", "p_lag"]).select([
    pl.col("match_key").alias("match"), "round", "second",
    "alive_ct", "alive_t",
    pl.col("st_equip_ct").alias("equip_ct"), pl.col("st_equip_t").alias("equip_t"),
    pl.col("bomb_planted").cast(pl.Int64).alias("planted"),
    pl.when(pl.col("bomb_planted") == 1).then(pl.col("seconds_since_plant")).otherwise(0.0).alias("since_plant"),
    pl.col("label_ct_win").cast(pl.Int64).alias("y"),
    "p_v6", "p_v4", "p_xgb83", "p_gnn_hyb1", "p_gnn_set", "p_gru", "p_lag",
])
out.write_parquet(OUT + "/rows_oct14_25.parquet")
print("WROTE", OUT + "/rows_oct14_25.parquet", "rows", out.height, "matches", out["match"].n_unique(),
      "rounds", out.select("match", "round").n_unique(), flush=True)

# quick pooled log loss on these rows, for orientation only (the grader's table-based score is the grade)
def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


y = out["y"].to_numpy().astype(np.float64)
for c in ["p_v6", "p_v4", "p_xgb83", "p_gnn_hyb1", "p_gnn_set", "p_gru", "p_lag"]:
    print("pooled LL", c, round(ll(out[c].to_numpy().astype(np.float64), y), 5), flush=True)
