import json, sys
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
M = pl.read_parquet(r"G:\pureskill\csds-cs2\data\combined\matches.parquet", columns=["match_key", "upload_day", "in_scope", "dup_keeper", "map_name"])
S = json.load(open(r"G:\pureskill\csds-cs2\data\combined\split_manifest.json", encoding="utf-8"))
assign = S.get("assign", S)
print("manifest keys:", len(assign), "sample values:", list(set(assign.values()))[:5])
M = M.filter(pl.col("in_scope") & pl.col("dup_keeper"))
M = M.with_columns(pl.col("match_key").replace_strict(assign, default=None, return_dtype=pl.Utf8).alias("split"))
M = M.with_columns(pl.col("upload_day").cast(pl.Utf8).str.replace_all("/", "-").str.slice(0, 10).alias("ud"))
def period(ud):
    return pl.when(pl.col("ud") < "2026-10-08").then(pl.lit("a) before Oct 8")).when(pl.col("ud") < "2026-10-09").then(pl.lit("b) Oct 8")).when(pl.col("ud") < "2026-10-26").then(pl.lit("c) Oct 9-25")).when(pl.col("ud") <= "2026-10-31").then(pl.lit("d) Oct 26-31")).otherwise(pl.lit("e) after Oct 31"))
M = M.with_columns(period(None).alias("period"))
t = M.group_by(["split", "period"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("split") * 100).round(1).alias("pct")).sort(["split", "period"])
print(t)
print("holdout uploads by day (Oct 26-31):")
print(M.filter(pl.col("period") == "d) Oct 26-31").group_by("ud").len().sort("ud"))
