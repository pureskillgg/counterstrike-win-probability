import json, collections, polars as pl, sys
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))["assign"]
caches={"base":G+"/cache_snapshot_2026.parquet","ext":G+"/cache_snapshot_2026_ext.parquet","exp":G+"/cache_snapshot_2026_oct23_31.parquet"}
# sample keys
k0=pl.scan_parquet(caches["base"]).select("match_key").head(3).collect()["match_key"].to_list()
print("cache match_key sample", k0)
print("manifest key sample", list(m.keys())[:2])
