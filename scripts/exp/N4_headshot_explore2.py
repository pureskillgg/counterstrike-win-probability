import glob
import os

import polars as pl

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
STATE = "G:/pureskill/csds-cs2/data/state_sample"


def pages(channel):
    folder = os.path.join(TOME, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)

# (a) era split
ph_cols = ["match_key", "round", "tick", "player_id", "attacker_id", "health", "weapon_name",
           "health_removed", "hit_box_code", "player_team_code", "attacker_team_code",
           "player_x_pos", "player_y_pos", "player_z_pos", "attacker_x_pos", "attacker_y_pos",
           "attacker_z_pos", "player_id_fixed", "attacker_id_fixed"]
hurt = pl.scan_parquet(pages("player_hurt"), missing_columns="insert").select(ph_cols).collect()
print("player_hurt all rows:", hurt.height)
print(hurt.with_columns(pl.col("match_key").str.slice(0, 10).alias("era")).group_by("era").agg(
    pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches")))

h26 = hurt.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("2026 player_hurt rows:", h26.height, "matches:", h26.select("match_key").n_unique())

# (b) exact weapon names in 2026
print("\n2026 weapon_name values:")
print(h26.group_by("weapon_name").agg(pl.len().alias("rows")).sort("rows", descending=True))

# (c) hit_box_code vs mean health_removed for rifles
rifles = h26.filter(pl.col("weapon_name").str.contains("ak|m4", literal=False))
print("\nrifle hits by hit_box_code (2026):")
print(rifles.group_by("hit_box_code").agg(
    pl.len().alias("rows"), pl.col("health_removed").mean().alias("mean_health_removed")).sort("hit_box_code"))
print("\nall hits by hit_box_code (2026):")
print(h26.group_by("hit_box_code").agg(
    pl.len().alias("rows"), pl.col("health_removed").mean().alias("mean_health_removed")).sort("hit_box_code"))

# (e) duplicates
key = ["match_key", "round", "tick", "player_id", "attacker_id", "hit_box_code", "health", "health_removed", "weapon_name"]
print("\n2026 exact-key duplicate rows (match,round,tick,victim,attacker,box,health,hr,weapon):",
      h26.height - h26.unique(subset=key).height)

# (d) round_end player_count
re_cols = ["match_key", "round", "winner_team_code", "player_count", "player_count_norm"]
re_ = pl.scan_parquet(pages("round_end"), missing_columns="insert").select(re_cols).collect()
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("\nround_end 2026 rows:", re26.height, "matches:", re26.select("match_key").n_unique())
print(re26.group_by(["player_count", "player_count_norm"]).agg(pl.len().alias("rows")).sort("rows", descending=True))

# (f) player_death weapon names 2026
pd_cols = ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "is_headshot"]
pdeath = pl.scan_parquet(pages("player_death"), missing_columns="insert").select(pd_cols).collect()
pd26 = pdeath.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("\nplayer_death 2026 rows:", pd26.height, "matches:", pd26.select("match_key").n_unique())
print(pd26.group_by("weapon_name").agg(pl.len().alias("rows"), pl.col("is_headshot").sum().alias("headshots")).sort("rows", descending=True))

# (g) state sample
p1 = pl.read_parquet(os.path.join(STATE, "players_1hz.parquet"), columns=[
    "match_key", "round", "tick", "seconds_since_live", "split", "player_id", "team_code", "health",
    "has_helmet", "equipment_value_calc"])
print("\nplayers_1hz rows:", p1.height)
print(p1.with_columns(pl.col("match_key").str.slice(0, 10).alias("era")).group_by("era").agg(
    pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches")))
print("team_code values:", p1.group_by("team_code").agg(pl.len()).sort("team_code").to_dicts())
print("player_id range:", p1["player_id"].min(), p1["player_id"].max())
print("tick mod 64 distribution (first 10):", p1.with_columns((pl.col("tick") % 64).alias("m")).group_by("m").agg(pl.len()).sort("m").head(10).to_dicts())
print("seconds_since_live min/max:", p1["seconds_since_live"].min(), p1["seconds_since_live"].max())
print("seconds_since_live==0 rows:", p1.filter(pl.col("seconds_since_live") == 0).height)
print("has_helmet value counts:", p1.group_by("has_helmet").agg(pl.len()).to_dicts())
print("split values:", p1.group_by("split").agg(pl.n_unique("match_key").alias("matches")).to_dicts())

sample26 = set(p1.filter(pl.col("match_key").str.starts_with("csds/2026/"))["match_key"].unique().to_list())
tome26 = set(h26["match_key"].unique().to_list())
print("\nstate sample matches (2026):", len(sample26), " tome 2026 hurt matches:", len(tome26),
      " overlap:", len(sample26 & tome26))
print("sample 2026 match example:", sorted(sample26)[:2])
print("tome 2026 match example:", sorted(tome26)[:2])

# per-match rounds in state sample vs tome
rounds = pl.read_parquet(os.path.join(STATE, "rounds.parquet"))
print("\nrounds.parquet 2026 matches:", rounds.filter(pl.col("match_key").str.starts_with("csds/2026/")).select("match_key").n_unique())
print(rounds.head(3))
