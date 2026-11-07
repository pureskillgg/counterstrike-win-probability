# COUNTS ONLY. No key strings or holdout content are printed.
import json, polars as pl
OUTD = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/"
FIN = "/home/sagemaker-user/adam/builder/final/"
km = pl.read_parquet(OUTD + "key_chunk_map.parquet")
mem = pl.read_parquet(OUTD + "membership_long.parquet").filter(pl.col("kind") == "status")
export_dates = km["date"]
print("export date range (min, max) and count dated >= 2026-10-26:", export_dates.min(), export_dates.max(), km.filter(pl.col("date") >= "2026-10-26").height)
hb = set(json.load(open("/home/sagemaker-user/adam_holdout_b_keys.json"))["match_keys"])
ex = set(km["match_key"].to_list())
allchunk = set(mem["match_key"].to_list())
print("holdout B file keys:", len(hb), "| in export key list:", len(hb & ex), "| in any S3 chunk c000-c067:", len(hb & allchunk))
m = json.load(open(FIN + "split_manifest_v2.json"))
print("manifest_v2 fields:", {k: type(v).__name__ for k, v in m.items()})
a = m["assign"]
print("assign type:", type(a).__name__, "len", len(a))
print("frozen_from:", m.get("frozen_from"), "| rule:", str(m.get("rule"))[:400])
print("history entries:", len(m.get("history", [])))
