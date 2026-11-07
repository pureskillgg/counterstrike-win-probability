# COUNTS ONLY. Reads only the match_key column of the local 1 Hz files. No feature or label values are read.
import glob, json, polars as pl, os, time
t0 = time.time()
OUTD = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/"
LOC = "/home/sagemaker-user/adam/builder/laptop_status/status_1hz/"
files = sorted(glob.glob(LOC + "*.parquet"))
print("local files", len(files), "first", os.path.basename(files[0]), "last", os.path.basename(files[-1]))
sch = pl.read_parquet_schema(files[0])
print("schema names (first 12):", list(sch.keys())[:12])
kcol = "match_key" if "match_key" in sch else None
print("match_key column present:", kcol is not None)
loc_keys = {}
for f in files:
    d = os.path.basename(f).replace(".parquet", "")
    ks = set(pl.read_parquet(f, columns=[kcol])[kcol].unique().to_list())
    loc_keys[d] = ks
allk = set().union(*loc_keys.values())
print("distinct local match keys across files:", len(allk), "| seconds", round(time.time() - t0, 1))
km = pl.read_parquet(OUTD + "key_chunk_map.parquet")
def cnt(filter_expr, label):
    sub = set(km.filter(filter_expr)["match_key"].to_list())
    print(label, "keys", len(sub), "| present in local 1Hz:", len(sub & allk))
cnt((pl.col("reason") == "test_split_coverage_only") & (pl.col("n_status_chunks") == 0), "test keys in no S3 chunk")
cnt((pl.col("reason") == "test_split_coverage_only") & (pl.col("n_status_chunks") > 0), "test keys in S3 chunks")
cnt((pl.col("reason") == "local_1hz_only_not_in_channel") & (pl.col("n_status_chunks") == 0), "local-only built keys in no S3 chunk")
cnt(pl.col("reason") == "protected_date_2026-10-21_onward", "protected 2848 keys")
# per-day: how many of the 1,164 no-chunk test keys each file holds (counts by day, for sizing)
t_no = set(km.filter((pl.col("reason") == "test_split_coverage_only") & (pl.col("n_status_chunks") == 0))["match_key"].to_list())
per_day = {d: len(ks & t_no) for d, ks in loc_keys.items() if len(ks & t_no)}
print("no-chunk test keys: days with any presence", len(per_day), "| total key-day pairs", sum(per_day.values()))
print("sample of day counts (first 5 and last 5):", list(per_day.items())[:5], list(per_day.items())[-5:])
json.dump({"local_files": len(files), "distinct_local_keys": len(allk), "no_chunk_test_keys_in_local": len(t_no & allk)},
          open(OUTD + "local_membership_counts.json", "w"), indent=1)
