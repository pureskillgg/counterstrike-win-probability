# COUNTS ONLY. Prints label counts and set sizes. No key strings are printed.
import json, polars as pl, collections
OUTD = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/"
FIN = "/home/sagemaker-user/adam/builder/final/"
km = pl.read_parquet(OUTD + "key_chunk_map.parquet")
mem = pl.read_parquet(OUTD + "membership_long.parquet").filter(pl.col("kind") == "status")
m = json.load(open(FIN + "split_manifest_v2.json"))
assign = m["assign"]
vals = collections.Counter(type(v).__name__ + ":" + str(v) for v in assign.values())
print("assign value counts (all manifest keys):", dict(vals.most_common(10)))
hb = set(json.load(open("/home/sagemaker-user/adam_holdout_b_keys.json"))["match_keys"])
prot = set(km.filter(pl.col("reason") == "protected_date_2026-10-21_onward")["match_key"].to_list())
test = set(km.filter(pl.col("split") == 2)["match_key"].to_list())
def cnt(keyset):
    return dict(collections.Counter(str(assign.get(k, "MISSING")) for k in keyset))
print("holdout B keys: manifest label counts", cnt(hb))
print("protected 2848 keys: manifest label counts", cnt(prot))
print("test-split 5677 keys: manifest label counts", cnt(test))
hb_chunks = mem.filter(pl.col("match_key").is_in(list(hb))).group_by("chunk").agg(pl.len().alias("n")).sort("chunk")
print("holdout B keys per chunk (count):", hb_chunks.to_dicts())
print("export keys dated Oct21-25 by manifest label:", cnt(set(km.filter(pl.col("date") >= "2026-10-21")["match_key"].to_list())))
