"""For the 11 holdout matches in duplicate groups: the split of each group's other members in the current manifest. Keys and splits only."""
import json, os
import polars as pl
ROOT = r"G:\pureskill\csds-cs2"
cur = json.load(open(os.path.join(ROOT, "data", "combined", "split_manifest.json"), encoding="utf-8"))["assign"]
H = set(open(os.path.join(ROOT, "data", "holdout_b", "holdout_keys.txt"), encoding="utf-8").read().split())
M = pl.read_parquet(os.path.join(ROOT, "data", "combined", "matches.parquet"), columns=["match_key", "dup_group", "dup_group_size", "dup_keeper", "map_name"])
hg = M.filter(pl.col("match_key").is_in(list(H)) & pl.col("dup_group").is_not_null() & (pl.col("dup_group_size") > 1))
groups = hg["dup_group"].unique().to_list()
mem = M.filter(pl.col("dup_group").is_in(groups)).select("match_key", "dup_group", "map_name", "dup_keeper")
mem = mem.with_columns(pl.col("match_key").map_elements(lambda k: "holdout" if k in H else cur.get(k, "absent"), return_dtype=pl.Utf8).alias("split"))
print("groups", len(groups), "members", mem.height)
print(mem.group_by("split").len().to_dicts())
print("members per group:", mem.group_by("dup_group").len()["len"].value_counts().sort("len").rows())
print("partner split counts (non-holdout members):", mem.filter(pl.col("split") != "holdout").group_by("split").len().to_dicts())
print("maps of groups:", mem.group_by("map_name").len().to_dicts())
