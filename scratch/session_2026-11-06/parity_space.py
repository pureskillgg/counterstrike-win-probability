"""Parity on the space: bucket features (pos_parity) vs laptop features (laptop_pos) for the same matches, seconds and players."""
import glob

import polars as pl

HOME = "/home/sagemaker-user/adam/builder"
sp = pl.concat([pl.read_parquet(f"{HOME}/final/pos_parity/bucket_{c}.parquet") for c in ["c050", "c060"]])
keys = sp.select("match_key").unique()
lap = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{HOME}/laptop_pos/every2_sites_v1/*.parquet"))])
lap = lap.join(keys, on="match_key", how="semi")
feats = [c for c in sp.columns if c not in ("match_key", "round", "tick")]
print("space rows", sp.height, "matches", sp["match_key"].n_unique(), "| laptop rows", lap.height, "matches", lap["match_key"].n_unique(), flush=True)
j = sp.join(lap.select(["match_key", "round", "tick"] + [c for c in feats if c in lap.columns]),
            on=["match_key", "round", "tick"], how="full", suffix="_lap", coalesce=True)
print("space-only", int(j.filter(pl.col("c_site_hi_lap").is_null() & pl.col("c_site_hi").is_not_null()).height),
      "| laptop-only", int(j.filter(pl.col("c_site_hi").is_null() & pl.col("c_site_hi_lap").is_not_null()).height), flush=True)
both = j.filter(pl.col("c_site_hi").is_not_null() & pl.col("c_site_hi_lap").is_not_null())
print("joined rows", both.height, flush=True)
for c in feats:
    a = both[c].cast(pl.Float64)
    b = both[c + "_lap"].cast(pl.Float64)
    nd = int((a.is_null() != b.is_null()).sum())
    mx = float(((a - b).abs().filter(a.is_not_null() & b.is_not_null())).max() or 0.0)
    print(f"{c:24s} null-mismatch {nd:8d}  max|diff| {mx:.4g}", flush=True)
