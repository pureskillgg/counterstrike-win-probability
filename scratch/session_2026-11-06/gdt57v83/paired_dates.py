import json, collections
import numpy as np, polars as pl
OUT = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/gdt57v83"
V57 = pl.read_parquet(f"{OUT}/val_rescored_v6_drop_heavy_57.parquet")
P83 = pl.read_parquet("G:/pureskill/csds-cs2/data/sprint/run/gpu_fast/preds/full_d4_e0.03_m20_s0.parquet")
assert V57.select(["match_key", "round", "second", "label_ct_win"]).equals(P83.select(["match_key", "round", "second", "label_ct_win"]))
keys = V57["match_key"].to_list()
dates = sorted({"-".join(k.split("/")[1:4]) for k in set(keys)})
per_match_date = collections.Counter("-".join(k.split("/")[1:4]) for k in set(keys))
res = {"validation_match_date_min": dates[0], "validation_match_date_max": dates[-1], "distinct_dates": len(dates),
       "matches_per_date_min": min(per_match_date.values()), "matches_per_date_max": max(per_match_date.values()),
       "key_prefixes": dict(collections.Counter(k.split("/")[0] + "/" + k.split("/")[1] for k in set(keys)))}
y = V57["label_ct_win"].to_numpy().astype(np.float64)
def ll(p):
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
d = ll(P83["p"].cast(pl.Float64).to_numpy()) - ll(V57["p"].to_numpy())
uniq, inv = np.unique(np.asarray(keys), return_inverse=True)
s = np.bincount(inv, weights=d); c = np.bincount(inv).astype(np.float64)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(s), size=(1000, len(s)))
means = s[idx].sum(axis=1) / c[idx].sum(axis=1)
res["paired_validation_83_minus_57_logloss"] = {"mean": float(s.sum() / c.sum()), "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    "note": "negative = 83 candidate better; match-cluster bootstrap 1000 resamples seed 0; validation rows only"}
res["logloss_83_validation"] = float(ll(P83["p"].cast(pl.Float64).to_numpy()).mean())
res["logloss_57_validation"] = float(ll(V57["p"].to_numpy()).mean())
json.dump(res, open(f"{OUT}/paired_dates_val.json", "w", encoding="utf-8"), indent=2)
print(json.dumps(res, indent=2), flush=True)
