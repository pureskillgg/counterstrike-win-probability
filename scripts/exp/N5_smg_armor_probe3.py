import polars as pl
pl.Config.set_tbl_rows(300)
T = r"G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/"
R = r"G:/pureskill/csds-cs2/data/state_sample/rounds.parquet"
B = r"G:/pureskill/csds-cs2/data/state_sample/round_buy_types.parquet"
codes = [7,8,9,10,11,13,14,16,17,19,23,24,25,26,27,28,29,33,34,35,38,39,40,60]
lf = pl.scan_parquet(T + "basic_weapon_fire.COMBINED/dataframe_*")
res = (lf.select(["match_key","player_weapon_code","weapon_name"])
         .filter(pl.col("match_key").str.starts_with("csds/2026/"))
         .filter(pl.col("player_weapon_code").is_in(codes))
         .group_by(["player_weapon_code","weapon_name"]).agg(pl.len().alias("rows"))
         .sort(["player_weapon_code","rows"], descending=[False, True]).collect())
for r in res.iter_rows():
    print("weapon_fire", r)
rd = pl.read_parquet(R)
print("rounds round min/max:", rd.select(pl.col("round").min().alias("mn"), pl.col("round").max().alias("mx")).row(0))
print("rounds round_number min/max:", rd.select(pl.col("round_number").min().alias("mn"), pl.col("round_number").max().alias("mx")).row(0))
print("round==round_number:", rd.filter(pl.col("round") == pl.col("round_number")).height, "of", rd.height)
print("rounds split:", rd.group_by("split").agg(pl.len().alias("n"), pl.col("match_key").n_unique().alias("m")).to_dicts())
bt = pl.read_parquet(B)
pist = bt.filter(pl.col("is_pistol"))
print("is_pistol rows:", pist.height, "distinct round values:", sorted(pist.select("round").unique().to_series().to_list()))
print("is_pistol by round:", pist.group_by("round").agg(pl.len().alias("n")).sort("round").to_dicts())
print("round_buy_types rows:", bt.height, "match count:", bt.select(pl.col("match_key").n_unique()).item())
print("rows where (round in {1,13}) != is_pistol:", bt.filter(pl.col("round").is_in([1,13]) != pl.col("is_pistol")).height)
