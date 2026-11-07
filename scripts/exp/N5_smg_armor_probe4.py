import polars as pl
T = r"G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/"
R = r"G:/pureskill/csds-cs2/data/state_sample/rounds.parquet"
B = r"G:/pureskill/csds-cs2/data/state_sample/round_buy_types.parquet"
codes = [8,9,10,11,13,14,16,17,19,23,24,25,26,27,28,29,33,34,35,38,39,40,60]
lf = pl.scan_parquet(T + "basic_weapon_fire.COMBINED/dataframe_*")
res = (lf.select(["match_key","player_weapon_code","weapon_name"])
         .filter(pl.col("match_key").str.starts_with("csds/2026/"))
         .filter(pl.col("player_weapon_code").is_in(codes))
         .group_by(["player_weapon_code","weapon_name"]).agg(pl.len().alias("rows"))
         .collect())
tot = res.group_by("player_weapon_code").agg(pl.col("rows").sum().alias("tot"))
top = (res.sort(["player_weapon_code","rows"], descending=[False,True])
          .group_by("player_weapon_code", maintain_order=True).first()
          .join(tot, on="player_weapon_code"))
for r in top.sort("player_weapon_code").iter_rows(named=True):
    print("code", r["player_weapon_code"], "top_name", r["weapon_name"], "rows", r["rows"], "of", r["tot"], "share", round(r["rows"]/r["tot"],4))
rd = pl.read_parquet(R)
print("rounds round min/max:", rd["round"].min(), rd["round"].max(), "round==round_number:", rd.filter(pl.col("round")==pl.col("round_number")).height, "of", rd.height)
bt = pl.read_parquet(B)
pist = bt.filter(pl.col("is_pistol"))
print("is_pistol rows:", pist.height)
print("is_pistol by round:", sorted(pist.group_by("round").agg(pl.len().alias("n")).iter_rows()))
print("disagree (round in {1,13}) vs is_pistol:", bt.filter(pl.col("round").is_in([1,13]) != pl.col("is_pistol")).height)
