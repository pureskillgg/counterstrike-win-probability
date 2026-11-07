import sys
import polars as pl
import numpy as np

print("python", sys.version.split()[0])
print("polars", pl.__version__, "numpy", np.__version__)

BASE = r"G:\pureskill\csds-cs2\data\state_sample"

p = pl.scan_parquet(BASE + r"\players_1hz.parquet")
r = pl.scan_parquet(BASE + r"\rounds.parquet")
rb = pl.scan_parquet(BASE + r"\round_buy_types.parquet")

print("\n== players_1hz schema ==")
for k, v in p.collect_schema().items():
    print(f"  {k}: {v}")

print("\n== rounds schema ==")
for k, v in r.collect_schema().items():
    print(f"  {k}: {v}")

print("\n== round_buy_types schema ==")
for k, v in rb.collect_schema().items():
    print(f"  {k}: {v}")

print("\n== rounds head ==")
with pl.Config(tbl_cols=20, tbl_width_chars=250):
    print(r.head(5).collect())

print("\n== round_buy_types head ==")
with pl.Config(tbl_cols=20, tbl_width_chars=250):
    print(rb.head(5).collect())

print("\n== players_1hz head ==")
with pl.Config(tbl_cols=30, tbl_width_chars=300):
    print(p.head(12).collect())

print("\n== seconds_since_live range and snapshot-0 counts ==")
s0 = p.filter(pl.col("seconds_since_live") == 0)
print(
    s0.select(
        pl.len().alias("rows_at_s0"),
        pl.col("seconds_since_live").min().alias("min_ssl"),
        pl.col("seconds_since_live").max().alias("max_ssl"),
    ).collect()
)
print(
    p.select(pl.col("seconds_since_live").min().alias("min"), pl.col("seconds_since_live").max().alias("max"), pl.len().alias("n")).collect()
)

print("\n== team_code and team counts at s0 ==")
print(
    s0.group_by("team_code")
    .agg(pl.len().alias("n_players"))
    .sort("team_code")
    .collect()
)

print("\n== per (match_key, round, team) alive count at s0 distribution ==")
per_team = (
    s0.group_by(["match_key", "round", "team_code"])
    .agg(pl.len().alias("n"))
    .collect()
)
print(per_team.group_by(["team_code", "n"]).agg(pl.len().alias("rounds")).sort(["team_code", "n"]))

print("\n== distinct map_name in rounds ==")
print(r.group_by("map_name").agg(pl.len().alias("n")).sort("n", descending=True).collect())

print("\n== rounds split and ct_win counts ==")
print(r.group_by("split").agg(pl.len().alias("n"), pl.col("ct_win").mean().alias("ct_win_mean")).collect())
