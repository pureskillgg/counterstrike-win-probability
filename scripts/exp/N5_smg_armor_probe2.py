import polars as pl
pl.Config.set_tbl_rows(200)
P = r"G:/pureskill/csds-cs2/data/state_sample/players_1hz.parquet"
R = r"G:/pureskill/csds-cs2/data/state_sample/rounds.parquet"
B = r"G:/pureskill/csds-cs2/data/state_sample/round_buy_types.parquet"
p0 = pl.scan_parquet(P).filter(pl.col("seconds_since_live") == 0)
d0 = p0.collect()
print("inv_primary full list at sec0:")
for r in d0.group_by("inv_primary").agg(pl.len().alias("n")).sort("inv_primary").iter_rows():
    print(r)
print("money stats:", d0.select(pl.col("money").mean().alias("mean"), pl.col("money").min().alias("min"), pl.col("money").max().alias("max")).row(0))
print("armor>=1 share:", d0.select((pl.col("armor") >= 1).mean()).row(0))
print("round max in players_1hz (any sec):", pl.scan_parquet(P).select(pl.col("round").max()).collect().item())
rd = pl.read_parquet(R)
print("rounds.parquet round range:", rd.select(pl.col("round").min(), pl.col("round").max()).row(0), "round_number range:", rd.select(pl.col("round_number").min(), pl.col("round_number").max()).row(0))
print("round==round_number rows:", rd.filter(pl.col("round") == pl.col("round_number")).height, "of", rd.height)
print("rounds per split:", rd.group_by("split").agg(pl.len().alias("n"), pl.col("match_key").n_unique().alias("m")).to_dicts())
bt = pl.read_parquet(B)
print("is_pistol rounds:")
print(bt.filter(pl.col("is_pistol")).group_by("round").agg(pl.len().alias("n")).sort("round").to_dicts())
print("is_pistol vs round in {1,13} agreement:", bt.select(((pl.col("round").is_in([1,13])) == pl.col("is_pistol")).mean()).row(0))
print("rows in round_buy_types not matching round in {1,13} & is_pistol:", bt.filter((pl.col("round").is_in([1,13])) != pl.col("is_pistol")).height)
