"""Era, scope and match date of the duplicate partners (not in the split manifest) of the holdout duplicate groups. Metadata only."""
import json, os
import polars as pl
ROOT = r"G:\pureskill\csds-cs2"
cur = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
H = set(open(os.path.join(ROOT, "data", "holdout_b", "holdout_keys.txt"), encoding="utf-8").read().split())
M = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "dup_group", "dup_group_size", "era", "in_scope", "upload_day", "map_name", "match_date"])
hg = M.filter(pl.col("match_key").is_in(list(H)) & pl.col("dup_group").is_not_null() & (pl.col("dup_group_size") > 1))["dup_group"].unique().to_list()
p = M.filter(pl.col("dup_group").is_in(hg) & ~pl.col("match_key").is_in(list(H)))
print(p.select("match_key", "era", "in_scope", "upload_day", "map_name", "match_date").with_columns(pl.col("match_key").str.slice(0, 40)).to_dicts())
print("partner in manifest:", sum(k in cur for k in p["match_key"].to_list()))
print("in manifest overall: 29885; matches overall:", M.height, "era counts:", M.group_by("era").len().to_dicts())
print("partners' era:", p.group_by("era").len().to_dicts(), "in_scope:", p.group_by("in_scope").len().to_dicts())
