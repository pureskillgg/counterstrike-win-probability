import polars as pl
snap = pl.read_parquet("G:/pureskill/csds-cs2/data/combined/cache_snapshot_2024.parquet", columns=["match_key", "st_hp_ct", "st_hp_t", "alive_ct"])
print("rows", snap.height, "NaN st_hp_ct", snap.filter(pl.col("st_hp_ct").is_nan()).height, "null st_hp_ct", snap["st_hp_ct"].null_count())
print("NaN st_hp_t", snap.filter(pl.col("st_hp_t").is_nan()).height, "null st_hp_t", snap["st_hp_t"].null_count())
print("NaN alive_ct", snap["alive_ct"].null_count())
