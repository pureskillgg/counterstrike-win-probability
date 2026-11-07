import polars as pl
T = r"G:/pureskill/csds-cs2/tomes/combined_test/tome/csds/"
lf = pl.scan_parquet(T + "basic_item_equip.COMBINED/dataframe_*")
res = (lf.select(["match_key", "item", "weapon_type_code"])
         .filter(pl.col("match_key").str.starts_with("csds/2026/"))
         .group_by(["item", "weapon_type_code"]).agg(pl.len().alias("rows"))
         .sort(["weapon_type_code", "item"])
         .collect())
print(res)
print("items with >1 class code:")
print(res.group_by("item").agg(pl.len().alias("n_codes")).filter(pl.col("n_codes") > 1))
