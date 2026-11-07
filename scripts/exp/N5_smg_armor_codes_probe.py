import polars as pl
T = r"G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/"
codes = [11, 14, 29, 35, 38]
out = {}
for ch, cols in [("basic_item_pickup.COMBINED", ["match_key", "def_index", "item"]),
                 ("basic_item_dropped.COMBINED", ["match_key", "def_index", "item"])]:
    lf = pl.scan_parquet(T + ch + "/dataframe_*")
    sch = lf.collect_schema()
    use = [c for c in cols if c in sch.names()]
    res = (lf.select(use)
             .filter(pl.col("match_key").str.starts_with("csds/2026/"))
             .filter(pl.col("def_index").is_in(codes))
             .group_by(["def_index", "item"]).agg(pl.len().alias("rows"))
             .sort(["def_index", "rows"], descending=[False, True])
             .collect())
    print(ch, "->")
    print(res)
lf = pl.scan_parquet(T + "basic_weapon_fire.COMBINED/dataframe_*")
res = (lf.select(["match_key", "player_weapon_code", "weapon_name"])
         .filter(pl.col("match_key").str.starts_with("csds/2026/"))
         .filter(pl.col("player_weapon_code").is_in(codes))
         .group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("rows"))
         .sort(["player_weapon_code", "rows"], descending=[False, True])
         .collect())
print("weapon_fire ->")
print(res)
