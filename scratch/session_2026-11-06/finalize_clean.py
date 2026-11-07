"""Clean training set: the v1 5v5 set minus the Mirage test and holdout B. Writes only to EBS (final/).

Excluded = de_mirage matches marked 'test' in the frozen split manifest (split_manifest.json, from the laptop)
           union holdout B (holdout_b_keys.txt, 653 keys, all 2026-10).
Oct 26-31 (holdout) is already absent from the v1 set.
Outputs: final/clean_matches.parquet, final/clean_rounds.parquet, final/clean_grid.parquet, final/clean_report.json
"""
import json
import os

import polars as pl

F = os.path.expanduser("~/adam/builder/final")
assign = json.load(open(os.path.join(F, "split_manifest.json"), encoding="utf-8"))["assign"]
holdout_b = [l.strip() for l in open(os.path.join(F, "holdout_b_keys.txt"), encoding="utf-8") if l.strip()]

lm = pl.read_parquet(os.path.join(F, "laptop_matches.parquet"), columns=["match_key", "map_name"])
bm = pl.read_parquet(os.path.join(F, "bucket_matches.parquet"), columns=["match_key", "map_name"])
maps = pl.concat([lm, bm]).unique("match_key")
mp = dict(zip(maps["match_key"].to_list(), maps["map_name"].to_list()))
mirage_test = [k for k, v in assign.items() if v == "test" and mp.get(k) == "de_mirage"]
excluded = sorted(set(mirage_test) | set(holdout_b))
print("excluded keys:", len(excluded), "| mirage test", len(mirage_test), "| holdout B", len(holdout_b),
      "| overlap", len(set(mirage_test) & set(holdout_b)), flush=True)

M0 = pl.read_parquet(os.path.join(F, "training_matches_v1.parquet"))
R0 = pl.read_parquet(os.path.join(F, "training_rounds_v1.parquet"))
G0 = pl.read_parquet(os.path.join(F, "grid_training_v1.parquet"))
ex = pl.Series("ex", excluded)

M = M0.filter(~pl.col("match_key").is_in(excluded))
R = R0.filter(~pl.col("match_key").is_in(excluded))
G = G0.filter(~pl.col("match_key").is_in(excluded))
M.write_parquet(os.path.join(F, "clean_matches.parquet"))
R.write_parquet(os.path.join(F, "clean_rounds.parquet"))
G.write_parquet(os.path.join(F, "clean_grid.parquet"))

use_m = R.filter(pl.col("use"))["match_key"].n_unique()
use_r = int(R["use"].sum())
report = {
    "excluded_keys": len(excluded), "mirage_test": len(mirage_test), "holdout_b": len(holdout_b),
    "removed_from_v1_matches": M0.height - M.height,
    "usable_matches_before": M0.filter(pl.col("dup_keeper") & (pl.col("n_ok_rounds") > 0) & pl.col("in_scope")).height,
    "usable_matches_after": use_m, "usable_rounds_after": use_r,
    "match_rows_after": M.height, "round_rows_after": R.height, "grid_rows_after": G.height,
    "still_contains_excluded": R.filter(pl.col("match_key").is_in(excluded)).height,
    "oct26_31_present": R.filter(pl.col("match_key").str.contains(r"^csds/2026/10/(2[6-9]|3[01])/")).height,
    "by_src": R.filter(pl.col("use")).group_by("src").agg(pl.col("match_key").n_unique().alias("matches")).rows(),
}
json.dump(report, open(os.path.join(F, "clean_report.json"), "w"), indent=1)
print(json.dumps(report, indent=1), flush=True)
