"""Duplicate-recording check for the holdout: do any holdout matches share a dup_group with a frozen match, or with each other?
Uses matches.parquet dup_group columns only (no labels, no model output)."""
import json, os
import polars as pl
ROOT = r"G:\pureskill\csds-cs2"
man = json.load(open(os.path.join(ROOT, "data", "combined", "_backup_pre_22_26_20261027", "split_manifest.json"), encoding="utf-8"))["assign"]
H = set(open(os.path.join(ROOT, "data", "holdout_b", "holdout_keys.txt"), encoding="utf-8").read().split())
M = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "dup_group", "dup_group_size", "dup_keeper"])
M = M.with_columns(pl.col("match_key").is_in(list(H)).alias("holdout"), pl.col("match_key").is_in(list(man.keys())).alias("frozen"))
print("matches with dup_group_size > 1:", M.filter(pl.col("dup_group_size") > 1).height)
g = M.filter(pl.col("dup_group").is_not_null() & (pl.col("dup_group_size") > 1))
multi = g.group_by("dup_group").agg(pl.len().alias("n"), pl.col("holdout").sum().alias("n_holdout"), pl.col("frozen").sum().alias("n_frozen"),
                                     pl.col("dup_keeper").sum().alias("n_keepers"))
print("dup groups:", multi.height)
print("dup groups containing a holdout match:", multi.filter(pl.col("n_holdout") > 0).height,
      "| of which also a frozen match:", multi.filter((pl.col("n_holdout") > 0) & (pl.col("n_frozen") > 0)).height,
      "| holdout-only groups:", multi.filter((pl.col("n_holdout") > 0) & (pl.col("n_frozen") == 0)).height)
print("holdout matches in a dup group:", g.filter(pl.col("holdout")).height, "of 653")
print("holdout matches flagged dup_keeper:", M.filter(pl.col("holdout") & (pl.col("dup_keeper") == True)).height, "| holdout non-keepers:", M.filter(pl.col("holdout") & (pl.col("dup_keeper") == False)).height)
