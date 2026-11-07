# v4 build lists: not-built export keys split into those a S3 channel chunk holds and those no chunk holds.
# Reads only export metadata (match_key, split, date, build_status) and the chunk-membership map from the locate scan.
import json
import polars as pl

KCM = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/key_chunk_map.parquet"
SUM = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/summary.json"
V4S = "/home/sagemaker-user/adam/builder/final/export_clean_v4/scripts/"
LAST_DATE = "2026-10-25"

summ = json.load(open(SUM))
pc = summ["per_chunk"]
for i in range(0, 68):
    c = "c%03d" % i
    assert pc[c].get("status_pages", 0) >= 1 and pc[c].get("info_pages", 0) >= 1, (c, pc[c])
print("chunks c000-c067 each have status and info pages (summary.json)")

d = pl.read_parquet(KCM)
assert d.height == 57811, d.height
nb = d.filter(pl.col("build_status") == "not_built")
print("not_built keys", nb.height)
assert nb.height == 8525, nb.height
print(nb.group_by(["split", "reason"]).agg([
    pl.len().alias("keys"),
    (pl.col("n_status_chunks") > 0).sum().alias("keys_in_a_chunk"),
    (pl.col("n_status_chunks") == 0).sum().alias("keys_in_no_chunk")]).sort(["split", "reason"]))
print("date range of not_built keys", nb["date"].min(), nb["date"].max())
assert nb["date"].max() <= LAST_DATE
assert nb["n_status_chunks"].max() <= 1
s3 = sorted(nb.filter(pl.col("n_status_chunks") > 0)["match_key"].to_list())
loc = sorted(nb.filter(pl.col("n_status_chunks") == 0)["match_key"].to_list())
print("S3 chunk targets", len(s3), "| local-only targets", len(loc))
assert len(s3) == 7361 and len(loc) == 1164
ls = nb.filter(pl.col("n_status_chunks") == 0)
print("local-only targets by split", ls.group_by("split").agg(pl.len().alias("n")).to_dicts())
print("local-only targets by month", ls.with_columns(pl.col("date").str.slice(0, 7).alias("m")).group_by("m").agg(pl.len().alias("n")).sort("m").to_dicts())
s3df = nb.filter(pl.col("n_status_chunks") > 0)
print("S3 targets by split", s3df.group_by("split").agg(pl.len().alias("n")).to_dicts())
print("S3 targets by chunk-range", s3df.with_columns(pl.col("status_chunks").list.first().alias("c")).group_by("c").agg(pl.len().alias("n")).sort("c").select(["c", "n"]).head(3).to_dicts(), "...")
json.dump(s3, open(V4S + "s3_target_v4.json", "w"))
json.dump(loc, open(V4S + "local_target_v4.json", "w"))
print("written s3_target_v4.json and local_target_v4.json in", V4S)
