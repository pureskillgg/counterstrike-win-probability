# COUNTS ONLY. Prints sizes, structure names and intersection sizes. No key strings, no holdout content is printed.
import json, polars as pl, os
OUTD = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/"
FIN = "/home/sagemaker-user/adam/builder/final/"
km = pl.read_parquet(OUTD + "key_chunk_map.parquet")
# 1) local source of the test keys that no S3 chunk holds
t0 = km.filter((pl.col("reason") == "test_split_coverage_only") & (pl.col("n_status_chunks") == 0))
print("test keys in no S3 chunk: src/build_status counts", t0.group_by("build_status").agg(pl.len().alias("n")).to_dicts())
st = pl.read_csv("/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/match_loadout_status.csv", schema_overrides={"match_key": pl.String})
print("status csv src for test keys in no S3 chunk:", st.join(t0.select("match_key"), on="match_key").group_by("src").agg(pl.len().alias("n")).to_dicts())
# 2) protected group: counts by date window (Oct 21-25 vs Oct 26-31), key-membership level only
pg = km.filter(pl.col("reason") == "protected_date_2026-10-21_onward")
print("protected keys by split code:", pg.group_by("split").agg(pl.len().alias("n")).sort("split").to_dicts())
print("protected keys Oct21-25 count:", pg.filter(pl.col("date") <= "2026-10-25").height, " Oct26-31 count:", pg.filter(pl.col("date") >= "2026-10-26").height)
print("test split max date (count of keys dated >= 2026-10-21):", km.filter((pl.col("split") == 2) & (pl.col("date") >= "2026-10-21")).height)
# 3) holdout files: structure and intersection COUNTS with the protected group
prot = set(pg["match_key"].to_list())
test = set(km.filter(pl.col("split") == 2)["match_key"].to_list())
def keys_from_obj(o):
    out = set()
    def walk(x):
        if isinstance(x, str) and x.startswith("csds/"):
            out.add(x)
        elif isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(o)
    return out
for path in [FIN + "holdout_b_keys.txt", "/home/sagemaker-user/adam_holdout_b_keys.json", FIN + "split_manifest.json", FIN + "split_manifest_v2.json"]:
    if not os.path.exists(path):
        print(path, "MISSING"); continue
    if path.endswith(".txt"):
        ks = set(l.strip() for l in open(path, encoding="utf-8") if l.strip().startswith("csds/"))
        top = "txt lines of csds keys"
    else:
        o = json.load(open(path, encoding="utf-8"))
        top = ("dict keys: " + str(list(o.keys())[:12])) if isinstance(o, dict) else ("list len %d" % len(o))
        ks = keys_from_obj(o)
    print(os.path.basename(path), "| structure:", top, "| csds keys found:", len(ks),
          "| in protected 2848:", len(ks & prot), "| in test split:", len(ks & test))
