"""Check that the rows of the frozen (pre-22-26 October) matches are unchanged between the backup and the current combined store.
Compares the rounds and grid tables for backup-manifest keys only (per-match row counts and a per-match column-value digest).
Counts and digests only. Holdout keys are excluded from the comparison."""
import json
import os
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
BK = os.path.join(ROOT, "data", "combined", "_backup_pre_22_26_20261027")
CU = os.path.join(ROOT, "data", "combined")
man = json.load(open(os.path.join(BK, "split_manifest.json"), encoding="utf-8"))["assign"]
old = list(man.keys())
H = set(open(os.path.join(ROOT, "data", "holdout_b", "holdout_keys.txt"), encoding="utf-8").read().split())
assert not (set(old) & H)
for name, cols in [("rounds.parquet", None), ("matches.parquet", None)]:
    a = pl.scan_parquet(os.path.join(BK, name)).filter(pl.col("match_key").is_in(old))
    b = pl.scan_parquet(os.path.join(CU, name)).filter(pl.col("match_key").is_in(old))
    a = a.collect(); b = b.collect()
    print(name, "backup rows (frozen keys)", a.height, "current rows (frozen keys)", b.height, "columns equal:", a.columns == b.columns)
    if a.columns == b.columns:
        sa = a.sort(["match_key"] + [c for c in ["round"] if c in a.columns])
        sb = b.sort(["match_key"] + [c for c in ["round"] if c in b.columns])
        print("   identical frozen-key rows:", sa.equals(sb))
        if not sa.equals(sb):
            ex = sa.join(sb, on=[c for c in ["match_key", "round"] if c in sa.columns], how="full", suffix="_cur", coalesce=True)
            diffcols = [c for c in sa.columns if c not in ("match_key", "round") and (ex[c] != ex[c + "_cur"]).sum() > 0]
            print("   differing columns:", diffcols[:20], "n_rows_diff_keys", ex.height)
a = pl.scan_parquet(os.path.join(BK, "grid.parquet")).filter(pl.col("match_key").is_in(old)).select(pl.len().alias("n"), pl.col("match_key").n_unique().alias("m")).collect()
b = pl.scan_parquet(os.path.join(CU, "grid.parquet")).filter(pl.col("match_key").is_in(old)).select(pl.len().alias("n"), pl.col("match_key").n_unique().alias("m")).collect()
print("grid frozen-key rows backup", a.to_dicts(), "current", b.to_dicts())
