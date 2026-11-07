import json, collections, polars as pl
G="G:/pureskill/csds-cs2/data/combined"
def day(k):
    p=k.split("/"); return "-".join(p[1:4])
pos=pl.scan_parquet(G+"/cache_pos_2026_trainctr.parquet")
print("pos cols", pos.collect_schema().names()[:12])
keys=pos.select("match_key").unique().collect()["match_key"].to_list()
print("pos distinct matches", len(keys))
print("pos Oct21-25", sum(1 for k in keys if "2026-10-21"<=day(k)<="2026-10-25"), "Oct26-31", sum(1 for k in keys if day(k)>="2026-10-26"), "Sep1-Oct13", sum(1 for k in keys if "2026-09-01"<=day(k)<="2026-10-13"))
m=json.load(open(G+"/split_manifest.json"))
print("manifest frozen_from", m.get("frozen_from"))
print("manifest history", json.dumps(m.get("history"))[:900])
print("manifest seed", m.get("seed"), "rule", m.get("rule")[:200])
