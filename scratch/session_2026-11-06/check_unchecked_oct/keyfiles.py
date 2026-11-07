import json, collections, polars as pl, sys, os
G="G:/pureskill/csds-cs2/data/combined"
m=json.load(open(G+"/split_manifest.json"))["assign"]
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
exp = pl.scan_parquet(G+"/cache_snapshot_2026_oct23_31.parquet").select("match_key").unique().collect()["match_key"].to_list()
expset=set(exp)
ext = set(pl.scan_parquet(G+"/cache_snapshot_2026_ext.parquet").select("match_key").unique().collect()["match_key"].to_list())
base = set(pl.scan_parquet(G+"/cache_snapshot_2026.parquet").select("match_key").unique().collect()["match_key"].to_list())
miss=[k for k,v in m.items() if v=="train" and "2026-10-21"<=day(k)<="2026-10-25" and k not in expset]
print("manifest Oct21-25 train keys missing from exp cache:", miss)
print("  in ext cache?", [k in ext for k in miss])
print("  in base?", [k in base for k in miss])
# none keys in exp cache: dates by path
none=[k for k in expset if k not in m]
dd=collections.Counter(day(k) for k in none)
print("exp cache keys not in manifest:", len(none), "by path day:", sorted(dd.items())[:3], "...", sorted(dd.items())[-3:], "distinct days", len(dd))
print("  path days 10-21..10-25 among none:", sum(n for d,n in dd.items() if "2026-10-21"<=d<="2026-10-25"))
print("  path days <=10-20 among none:", sum(n for d,n in dd.items() if d<"2026-10-21"))
# frozen test, centre fit, val keys
ft=json.load(open("C:/Users/13144/Documents/pureskill/counterstrike-win-probability/data/benchmark/frozen_test_keys.json"))
print("frozen type", type(ft).__name__, (list(ft.keys())[:5] if isinstance(ft,dict) else len(ft)))
