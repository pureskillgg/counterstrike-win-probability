# Counts only: chunk and group breakdowns from the key-only membership map. No feature or label values are read.
import json, polars as pl
OUTD = "/home/sagemaker-user/adam/builder/fixes/loadout_chunk_locate/"
km = pl.read_parquet(OUTD + "key_chunk_map.parquet")
print("export keys", km.height, "| keys in 0 chunks", km.filter(pl.col("n_status_chunks") == 0).height,
      "| keys in >1 chunk", km.filter(pl.col("n_status_chunks") > 1).height)
print("keys where status-chunk count != info-chunk count:", km.filter(pl.col("n_status_chunks") != pl.col("n_info_chunks")).height)
cg = pl.read_csv(OUTD + "chunk_by_reason_counts.csv")
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(40)
cols = [c for c in cg.columns]
print(cols)
print(cg.with_columns(pl.sum_horizontal(pl.exclude("chunk")).alias("row_total")).filter(pl.sum_horizontal(pl.exclude("chunk")) > 0))
print("column totals:", {c: int(cg[c].sum()) for c in cols if c != "chunk"})
# unbuilt groups: where do they sit
for r in ["test_split_coverage_only", "protected_date_2026-10-21_onward", "local_1hz_only_not_in_channel"]:
    sub = km.filter(pl.col("reason") == r)
    has = sub.filter(pl.col("n_status_chunks") > 0)
    print(r, "keys", sub.height, "| in S3 chunk", has.height, "| in no S3 chunk", sub.height - has.height)
    if has.height:
        ch = has.with_columns(pl.col("status_chunks").list.first().alias("chunk")).group_by("chunk").agg(pl.len().alias("n")).sort("chunk")
        print("   chunk range:", ch["chunk"].min(), "..", ch["chunk"].max(), "| distinct chunks:", ch.height)
    print("   by src-like status:", sub.group_by("build_status").agg(pl.len().alias("n")).to_dicts())
# the 1,164 test keys in no S3 chunk: do they sit in local caches (src) and month mix (counts)
ts0 = km.filter((pl.col("reason") == "test_split_coverage_only") & (pl.col("n_status_chunks") == 0))
print("test keys in no S3 chunk, by month:", ts0.with_columns(pl.col("date").str.slice(0, 7).alias("m")).group_by("m").agg(pl.len().alias("n")).sort("m").to_dicts())
ts1 = km.filter((pl.col("reason") == "test_split_coverage_only") & (pl.col("n_status_chunks") > 0))
print("test keys in S3 chunks, by month:", ts1.with_columns(pl.col("date").str.slice(0, 7).alias("m")).group_by("m").agg(pl.len().alias("n")).sort("m").to_dicts())
lo = km.filter((pl.col("reason") == "local_1hz_only_not_in_channel") & (pl.col("n_status_chunks") == 0))
print("local-only keys in no S3 chunk, by month:", lo.with_columns(pl.col("date").str.slice(0, 7).alias("m")).group_by("m").agg(pl.len().alias("n")).sort("m").to_dicts())
