import glob
import os

import numpy as np
import polars as pl

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)


def pages(channel):
    folder = os.path.join(TOME, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


ph_cols = ["match_key", "round", "tick", "player_id", "attacker_id", "health", "weapon_name",
           "health_removed", "hit_box_code", "player_team_code", "attacker_team_code", "attacker_weapon_code"]
hurt = pl.scan_parquet(pages("player_hurt"), missing_columns="insert").select(ph_cols).collect()
h = hurt.filter(pl.col("match_key").str.starts_with("csds/2026/"))
h = h.sort(["match_key", "round", "player_id", "tick", "health"], descending=[False, False, False, False, True])
h = h.with_columns(
    pl.col("health").shift(1).over(["match_key", "round", "player_id"]).alias("prev_health"),
    pl.col("tick").shift(1).over(["match_key", "round", "player_id"]).alias("prev_tick"),
    pl.col("attacker_id").shift(1).over(["match_key", "round", "player_id"]).alias("prev_attacker"),
    pl.col("weapon_name").shift(1).over(["match_key", "round", "player_id"]).alias("prev_weapon"),
    pl.col("health_removed").shift(1).over(["match_key", "round", "player_id"]).alias("prev_hr"),
    pl.col("match_key").shift(1).over(["match_key", "round", "player_id"]).alias("prev_mk"),
)
nonl = h.filter((pl.col("health") > 0) & pl.col("prev_health").is_not_null())
nonl = nonl.with_columns((pl.col("health") - (pl.col("prev_health") - pl.col("health_removed"))).alias("delta"))
print("non-lethal rows with prev present:", nonl.height)
print("delta==0:", nonl.filter(pl.col("delta") == 0).height)
print("delta value counts (top 12):")
print(nonl.group_by("delta").agg(pl.len()).sort("len", descending=True).head(12))
nonl = nonl.with_columns((pl.col("tick") - pl.col("prev_tick")).alias("gap"))
print("delta==0 by same-tick vs later:")
print(nonl.with_columns((pl.col("gap") == 0).alias("same_tick")).group_by(["same_tick", pl.col("delta") == 0]).agg(pl.len()))
print("mismatch examples:")
print(nonl.filter(pl.col("delta") != 0).select(
    ["match_key", "round", "player_id", "tick", "prev_tick", "attacker_id", "prev_attacker", "weapon_name",
     "prev_weapon", "health", "prev_health", "health_removed", "prev_hr", "delta"]).head(12))

# proportion of mismatch by whether previous hit is the same attacker
print("mismatch by weapon of current hit:")
print(nonl.with_columns((pl.col("delta") != 0).alias("mm")).group_by("mm").agg(pl.len()))
print("mismatch rate per match (top 5 matches):")
mm_rate = nonl.with_columns((pl.col("delta") != 0).alias("mm")).group_by("match_key").agg(
    pl.len().alias("n"), pl.col("mm").mean().alias("rate")).sort("rate", descending=True)
print(mm_rate.head(5))
print("matches with mismatch rate > 0.5:", mm_rate.filter(pl.col("rate") > 0.5).height, "of", mm_rate.height)
print("rate quantiles:", mm_rate.select(pl.col("rate").quantile(0.1).alias("p10"), pl.col("rate").median().alias("p50"), pl.col("rate").quantile(0.9).alias("p90")).to_dicts())

# lethal rows: is prev_health within [1,100]?
lethal = h.filter(pl.col("health") == 0)
print("\nlethal rows:", lethal.height, " prev present:", lethal.filter(pl.col("prev_health").is_not_null()).height)
print("lethal prev_health distribution (top 10):")
print(lethal.group_by("prev_health").agg(pl.len()).sort("len", descending=True).head(10))
print("lethal prev_health > 100 or < 1:", lethal.filter((pl.col("prev_health") > 100) | (pl.col("prev_health") < 1)).height)

# M4 variant check: weapon_name 'm4a1' vs attacker_weapon_code (hurt)
print("\nm4a1 attacker_weapon_code share:")
m4 = h.filter(pl.col("weapon_name") == "m4a1")
print(m4.group_by("attacker_weapon_code").agg(pl.len()).sort("len", descending=True).head(5))

# death table: weapon_name vs attacker_weapon_code is not present; check death weapon names for m4 variants vs hurt code 60/16
pd_cols = ["match_key", "round", "tick", "player_id", "attacker_id", "weapon_name", "is_headshot", "attacker_weapon_code"]
pdeath = pl.scan_parquet(pages("player_death"), missing_columns="insert").select(pd_cols).collect()
pd26 = pdeath.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("\nplayer_death weapon names per attacker_weapon_code for m4 variants:")
print(pd26.filter(pl.col("weapon_name").is_in(["m4a1", "m4a1_silencer", "m4a1_silencer_off"]) | pl.col("attacker_weapon_code").is_in([16, 60]))
      .group_by(["weapon_name", "attacker_weapon_code"]).agg(pl.len()).sort(["weapon_name", "len"], descending=[False, True]).head(12))
