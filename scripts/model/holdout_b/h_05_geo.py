"""S2 step 5: straight-line distances (SN arm inputs) for the holdout round-state rows, from the TRAIN-ONLY site centres.

(a) The centres are refitted on the frozen training keys (data/combined/_backup_pre_22_26_20261027/split_manifest.json, value "train")
    and compared with data/geometry/site_centres_compare.json (train_only, rounded to 0.01). This also checks that the saved
    SN features were fitted on the frozen training keys.
(b) distances() from sitecheck_geo.py is applied to the holdout round-state X arrays with those centres.
Output: data/holdout_b/geo/<map>_ext.parquet (match_key, round, second, st0..st3). Torch environment (sitecheck_geo imports torch modules).
"""
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))
import sitecheck_geo as SG  # noqa: E402
import make_pos_features as MPF  # noqa: E402

HB = os.path.join(ROOT, "data", "holdout_b")
OUTD = os.path.join(HB, "geo")
os.makedirs(OUTD, exist_ok=True)
MAPS = ["de_mirage", "de_dust2", "de_ancient", "de_anubis", "de_inferno", "de_nuke", "de_cache"]
man = json.load(open(os.path.join(ROOT, "data", "combined", "_backup_pre_22_26_20261027", "split_manifest.json"), encoding="utf-8"))["assign"]
train_keys = [k for k, v in man.items() if v == "train"]
Hs = set(open(os.path.join(HB, "holdout_keys.txt"), encoding="utf-8").read().split())
print(f"frozen train keys {len(train_keys)}; holdout keys in frozen manifest: {sum(k in man for k in Hs)}", flush=True)
assert not (set(train_keys) & Hs), "holdout key found in frozen training set"
train_centres = MPF.site_centres("combined", match_keys=train_keys)
saved = json.load(open(os.path.join(ROOT, "data", "geometry", "site_centres_compare.json"), encoding="utf-8"))
worst = 0.0
for m in MAPS:
    d = np.abs(np.asarray(train_centres[m]) - np.asarray(saved[m]["train_only"])).max()
    worst = max(worst, float(d))
    print(f"{m}: refit train-only centres {np.round(train_centres[m], 2).tolist()} | max diff vs saved (rounded) {d:.4f}", flush=True)
print(f"worst centre difference vs saved train-only centres: {worst:.4f} game units (rounding limit 0.005)", flush=True)
if worst > 0.01:
    raise SystemExit("refitted centres do not match the saved train-only centres; stopping")
for m in MAPS:
    z = np.load(os.path.join(HB, "round_state", f"{m}_ext.npz"))
    X, mi, keys = z["X"], z["match_idx"], z["match_keys"]
    st = SG.distances(m, X, train_centres[m])
    df = pl.DataFrame({"match_key": keys[mi], "round": z["round"].astype(np.int64), "second": z["second"].astype(np.int64)})
    df = df.with_columns([pl.Series(f"st{i}", st[:, i]) for i in range(SG.N_DIST)])
    df.write_parquet(os.path.join(OUTD, f"{m}_ext.parquet"))
    print(f"{m}: {df.height:,} holdout rows, {df['match_key'].n_unique()} matches -> geo", flush=True)
    del X, mi, keys, st, df, z
print("done", flush=True)
