import glob, os
import polars as pl
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(220)
BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"

def scan(ch):
    files = sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.COMBINED", "dataframe_*")))
    return pl.scan_parquet(files)

drop = scan("item_dropped").select(["match_key","round","tick","player_id","entity_id","item","def_index","reason","player_team_code"]).collect()
print("item_dropped rows", drop.height, "matches", drop["match_key"].n_unique())
print(drop.group_by("reason").len().sort("len", descending=True))
print("drop item/def pairs:")
print(drop.group_by(["def_index","item"]).len().sort("def_index"))
print("drop days:")
print(drop.with_columns(pl.col("match_key").str.slice(10,5).alias("day")).group_by("day").agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rows")).sort("day"))
print("drop null team:", drop["player_team_code"].null_count(), "null tick:", drop["tick"].null_count(), "null round:", drop["round"].null_count())
print("player_id values:", sorted(drop["player_id"].unique().to_list())[:20])
print("team codes:", drop.group_by("player_team_code").len().to_dicts())
del drop

pk = scan("item_pickup").filter(pl.col("match_key").str.starts_with("csds/2026/")).select(["match_key","round","tick","player_id","item","def_index","is_silent","player_team_code"]).collect()
print("item_pickup 2026 rows", pk.height, "matches", pk["match_key"].n_unique())
print(pk.group_by(["def_index","item"]).len().sort("def_index"))
print("pickup days:")
print(pk.with_columns(pl.col("match_key").str.slice(10,5).alias("day")).group_by("day").agg(pl.col("match_key").n_unique().alias("matches"), pl.len().alias("rows")).sort("day"))
print("pickup null team:", pk["player_team_code"].null_count(), "null def:", pk["def_index"].null_count())
del pk

re = scan("round_end").filter(pl.col("match_key").str.starts_with("csds/2026/")).select(["match_key","round","tick","winner_team_code","player_count","player_count_norm"]).collect()
print("round_end 2026 rows", re.height, "matches", re["match_key"].n_unique())
print(re.group_by("player_count").len().sort("player_count"))
print("dup (match,round):", re.height - re.unique(["match_key","round"]).height)
print("days:", re.with_columns(pl.col("match_key").str.slice(10,5).alias("day")).group_by("day").agg(pl.col("match_key").n_unique().alias("matches")).sort("day").to_dicts())
