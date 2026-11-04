"""Build the Oct 26-31 holdout feature matrix under pin 1b7c86fd. Features only (plus the label-null row filter). No labels are joined here
and no predictions are made. Writes X (float32, the 83 columns in matrix_trainctr feats_all order) and the row keys."""
import json
import os

import numpy as np
import polars as pl

ROOT = "G:/pureskill/csds-cs2"
OCT = f"{ROOT}/data/combined/cache_snapshot_2026_oct23_31.parquet"
POS = f"{ROOT}/data/combined/cache_pos_2026_trainctr.parquet"
TRAIN = f"{ROOT}/data/combined/cache_snapshot_2026.parquet"
HB = f"{ROOT}/data/combined/holdout_22_25_test_keys.json"
MATRIX = "C:/Users/13144/sprint/matrix_trainctr"
OUT = "C:/Users/13144/sprint/holdout_oct26_31"
os.makedirs(OUT, exist_ok=True)

meta = json.load(open(f"{MATRIX}/meta.json"))
feats, base = meta["feats_all"], meta["base"]
pos_cols = feats[len(base):]
assert len(feats) == 83 and len(base) == 59 and len(pos_cols) == 24, "feature recipe is not the 83-input full set"

keys = pl.read_parquet("C:/Users/13144/sprint/holdout_oct26_31_keys.parquet")["match_key"].to_list()
assert len(keys) == len(set(keys)) == 3315, f"population is {len(set(keys))} matches, pin says 3315"
hb = set(json.load(open(HB))["match_keys"])
train_keys = set(pl.scan_parquet(TRAIN).select("match_key").unique().collect()["match_key"].to_list())
assert not (set(keys) & hb), "holdout B key in population"
assert not (set(keys) & train_keys), "training-pool key in population"

pos = (pl.scan_parquet(POS).select(["match_key", "round", "tick"] + pos_cols)
       .with_columns([pl.col(c).cast(pl.Float32) for c in pos_cols]))
G = (pl.scan_parquet(OCT).filter(pl.col("match_key").is_in(keys) & pl.col("label_ct_win").is_not_null())
     .select(["match_key", "round", "tick", "second"] + base)
     .with_columns([pl.col(c).cast(pl.Float32) for c in base])
     .join(pos, on=["match_key", "round", "tick"], how="inner")
     .collect())
print("holdout rows (labelled cache seconds with a positions row):", G.height, flush=True)
print("holdout matches:", G["match_key"].n_unique(), flush=True)
assert G.height == 4320801, f"row count {G.height} differs from the pinned 4,320,801"
assert G.select(["match_key", "round", "tick"]).unique().height == G.height, "duplicate keys after join"

X = G.select(feats).to_numpy().astype(np.float32)
nan = int(np.isnan(X).sum())
print("NaN cells:", nan, flush=True)
# Missing values are part of the validation construction too (XGBoost takes NaN as missing). Compare per-column NaN rates with a
# sample of the validation rows of the stored matrix, so a new NaN pattern would show.
Xm = np.load(f"{MATRIX}/X.npy", mmap_mode="r")
n_fit, n_es = meta["n_fit"], meta["n_es"]
Xv = np.asarray(Xm[n_fit + n_es: n_fit + n_es + 500000])
hold_rate = np.isnan(X).mean(axis=0)
val_rate = np.isnan(Xv).mean(axis=0)
for f, a, b in zip(feats, hold_rate, val_rate):
    if a > 0 or b > 0:
        print(f"  {f}: holdout NaN {a:.4f} | validation sample NaN {b:.4f}", flush=True)
assert not np.all(np.isnan(X), axis=0).any(), "a column is entirely missing in the holdout"
assert float(np.max(np.abs(hold_rate - val_rate))) < 0.02, "NaN pattern differs from the validation matrix by more than 2 points"
np.save(f"{OUT}/X.npy", X)
G.select(["match_key", "round", "tick", "second"]).write_parquet(f"{OUT}/keys.parquet")
json.dump({"feats_all": feats, "base": base, "rows": int(X.shape[0]), "matches": int(G["match_key"].n_unique()),
           "source_cache": os.path.basename(OCT), "pos": os.path.basename(POS), "pin": "1b7c86fd"},
          open(f"{OUT}/meta.json", "w"), indent=1)
print("wrote", OUT, X.shape, flush=True)
