import sys
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_fmt_str_lengths(40)
P = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M5\M5_rounds.parquet"
df = pl.read_parquet(P)
mt = df.group_by(["map_name","batch"]).agg(pl.len().alias("rounds"), pl.col("match_key").n_unique().alias("matches"), pl.col("ct_win").sum().alias("ct_wins"), (pl.col("ct_win").mean()*100).round(2).alias("ct_pct"))
print(mt.sort(["map_name","batch"]))
print()
wide = df.group_by(["map_name","batch"]).agg((pl.col("ct_win").mean()*100).round(1).alias("ct")).pivot(values="ct", index="map_name", on="batch", aggregate_function="first").sort("map_name")
print("CT% wide:"); print(wide)
print()
mm = df.group_by(["map_name","batch","match_key"]).agg(pl.len().alias("r"), pl.col("ct_win").mean().alias("ctm")).group_by(["map_name","batch"]).agg(pl.len().alias("matches"), pl.col("ctm").mean().mul(100).round(2).alias("mean_match_ct_pct"), pl.col("ctm").std().round(3).alias("sd_match"))
print("per-match CT share (equal weight):"); print(mm.sort(["map_name","batch"]))
print()
print("overall by batch:")
print(df.group_by("batch").agg(pl.len(), pl.col("match_key").n_unique(), (pl.col("ct_win").mean()*100).round(2)))
