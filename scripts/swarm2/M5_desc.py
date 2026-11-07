import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\M5\M5_rounds.parquet"
df = pl.read_parquet(P)
print("rounds", df.height, "matches", df["match_key"].n_unique())
print("\nmatches by batch x map (match-level)")
m = df.group_by(["match_key", "batch", "map_name"]).agg(pl.len().alias("r")).group_by(["batch", "map_name"]).agg(pl.len().alias("matches"), pl.col("r").sum().alias("rounds")).sort(["map_name", "batch"])
print(m.pivot(values="matches", index="map_name", on="batch", aggregate_function="sum").sort("map_name"))
print("\nCT win rate by batch x map (round-level)")
g = df.group_by(["map_name", "batch"]).agg(pl.len().alias("rounds"), pl.col("ct_win").sum().alias("ct_wins"), (pl.col("ct_win").mean()*100).round(2).alias("ct_pct")).sort(["map_name", "batch"])
print(g)
print("\nmap share of rounds by batch")
s = df.group_by(["batch", "map_name"]).agg(pl.len().alias("n")).with_columns((pl.col("n")/pl.col("n").sum().over("batch")*100).round(1).alias("pct")).sort(["map_name","batch"])
print(s.pivot(values="pct", index="map_name", on="batch", aggregate_function="sum").sort("map_name"))
print("\nmax_rounds by batch")
print(df.group_by(["batch","max_rounds"]).agg(pl.len()).sort(["batch","max_rounds"]))
print("\nrounds per match by batch")
rpm = df.group_by(["batch","match_key"]).agg(pl.len().alias("r")).group_by("batch").agg(pl.col("r").mean().round(2).alias("mean_rounds"), pl.col("match_key").n_unique().alias("matches"))
print(rpm)
