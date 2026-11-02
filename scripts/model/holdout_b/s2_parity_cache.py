"""S2 check: does the snapshot builder reproduce the frozen 57 input columns on VALIDATION matches?

Takes 25 validation matches (frozen backup manifest, split == "val", seed 0 sample), builds their snapshot rows with the same
make_snapshot_cache.py used for the holdout (store data/holdout_b/parity, status from data/status_1hz), and compares the 57
features with data/combined/cache_snapshot_2026_ext.parquet on (match_key, round, second). Validation rows only.
"""
import json
import os
import random
import runpy
import sys

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
PAR = os.path.join(ROOT, "data", "holdout_b", "parity")
os.makedirs(PAR, exist_ok=True)
man = json.load(open(os.path.join(ROOT, "data", "combined", "_backup_pre_22_26_20261027", "split_manifest.json"), encoding="utf-8"))["assign"]
val = sorted(k for k, v in man.items() if v == "val")
G0 = pl.scan_parquet(os.path.join(ROOT, "data", "combined", "grid.parquet")).filter(pl.col("era") == "2026").select("match_key").unique().collect()["match_key"].to_list()
pool = sorted(set(val) & set(G0))
sample = random.Random(0).sample(pool, 25)
pl.DataFrame({"match_key": sample}).write_csv(os.path.join(PAR, "sample_keys.csv"))
Sl = pl.Series("match_key", sample)
pl.scan_parquet(os.path.join(ROOT, "data", "combined", "grid.parquet")).filter(pl.col("match_key").is_in(Sl) & (pl.col("era") == "2026")).sink_parquet(os.path.join(PAR, "grid.parquet"))
pl.scan_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet")).filter(pl.col("match_key").is_in(Sl)).sink_parquet(os.path.join(PAR, "matches.parquet"))
pl.scan_parquet(os.path.join(ROOT, "data", "combined", "rounds.parquet")).filter(pl.col("match_key").is_in(Sl)).sink_parquet(os.path.join(PAR, "rounds.parquet"))
sys.argv = ["make_snapshot_cache.py", "--store", "holdout_b/parity", "--status-dir", os.path.join(ROOT, "data", "status_1hz"),
            "--out-name", "cache_snapshot_2026_parity.parquet"]
runpy.run_path(os.path.join(ROOT, "scripts", "model", "make_snapshot_cache.py"), run_name="__main__")
feats = json.load(open(os.path.join(ROOT, "models", "winprob_v6_drop_heavy.meta.json"), encoding="utf-8"))["features"]
K = ["match_key", "round", "second"]
new = pl.read_parquet(os.path.join(PAR, "cache_snapshot_2026_parity.parquet"), columns=K + feats)
old = (pl.scan_parquet(os.path.join(ROOT, "data", "combined", "cache_snapshot_2026_ext.parquet"), ) .filter(pl.col("match_key").is_in(Sl))
       .select(K + feats).unique(K, keep="first").collect())
j = new.join(old, on=K, how="full", suffix="_old", coalesce=True)
both = j.filter(pl.col(feats[0]).is_not_null() & pl.col(feats[0] + "_old").is_not_null())
out = {"sample_matches": len(sample), "rows_new": new.height, "rows_old": old.height, "rows_both": both.height,
       "rows_only_new": int(j.filter(pl.col(feats[0] + "_old").is_null()).height),
       "rows_only_old": int(j.filter(pl.col(feats[0]).is_null()).height), "features": len(feats), "max_abs_diff_by_feature": {}}
worst = 0.0
for f in feats:
    a = both[f].cast(pl.Float64).to_numpy()
    b = both[f + "_old"].cast(pl.Float64).to_numpy()
    d = np.nanmax(np.abs(a - b)) if a.size else 0.0
    nn = int(np.sum(np.isnan(a) != np.isnan(b)))
    out["max_abs_diff_by_feature"][f] = float(d) if not np.isnan(d) else None
    worst = max(worst, 0.0 if np.isnan(d) else float(d))
    if nn:
        out.setdefault("null_mismatch", {})[f] = nn
out["worst_abs_diff"] = worst
json.dump(out, open(os.path.join(PAR, "parity_result.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "max_abs_diff_by_feature"}, indent=1))
