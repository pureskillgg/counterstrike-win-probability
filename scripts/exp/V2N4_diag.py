import glob, os
import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"

def pages(channel):
    d = os.path.join(BASE, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

# ---- round_end (2026)
re_ = pl.concat([pl.read_parquet(p) for p in pages("round_end")], how="diagonal_relaxed")
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("round_end 2026 rows:", re26.height, " matches:", re26["match_key"].n_unique())
print("dup (match_key, round) rows in round_end 2026:", re26.height - re26.unique(subset=["match_key", "round"]).height)
print("full-row duplicates in round_end 2026:", re26.height - re26.unique().height)
print("player_count vs player_count_norm (2026):")
print(re26.group_by(["player_count", "player_count_norm"]).len().sort(["player_count", "player_count_norm"]).to_dicts())
clean = re26.filter(pl.col("player_count") == 10).select(["match_key", "round", "winner_team_code", "tick"]).unique(subset=["match_key", "round"])
print("clean rounds (player_count==10, unique key):", clean.height, " matches with >=1 clean:", clean["match_key"].n_unique())
per_m = re26.group_by("match_key").agg(pl.len().alias("nr"), pl.col("round").max().alias("maxr"), (pl.col("player_count") == 10).sum().alias("nclean"))
print("matches where every round_end row is clean:", per_m.filter(pl.col("nclean") == pl.col("nr")).height)
print("matches with nr==max_round==24 and all clean:", per_m.filter((pl.col("nclean") == pl.col("nr")) & (pl.col("maxr") == 24)).height)
print("matches with 24 rounds (maxr==24):", per_m.filter(pl.col("maxr") == 24).height)
print("matches with nr==24 and nclean==24:", per_m.filter((pl.col("nr") == 24) & (pl.col("nclean") == 24)).height)
print("matches with nclean>=1 and nr==nclean:", per_m.filter((pl.col("nclean") >= 1) & (pl.col("nr") == pl.col("nclean"))).height)
print("distinct round values with >24:", re26.filter(pl.col("round") > 24).height)

# ---- player_hurt (2026)
ph = pl.concat([pl.read_parquet(p) for p in pages("player_hurt")], how="diagonal_relaxed")
ph26 = ph.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("player_hurt 2026 rows:", ph26.height)
print("full-row duplicates in player_hurt 2026:", int(ph26.is_duplicated().sum()))
keycols = ["match_key", "round", "tick", "player_id", "attacker_id", "hit_box_code", "weapon_name", "health", "health_removed"]
print("duplicates on key cols (match_key, round, tick, player_id, attacker_id, hit_box_code, weapon_name, health, health_removed):", int(ph26.select(keycols).is_duplicated().sum()))
print("NULL counts in key cols:", {c: ph26[c].null_count() for c in ["round", "tick", "player_id", "health", "health_removed", "hit_box_code", "weapon_name", "attacker_x_pos", "player_x_pos"]})

# join clean rounds
ph26c = ph26.join(clean.select(["match_key", "round"]), on=["match_key", "round"], how="inner")
print("player_hurt 2026 rows in clean rounds:", ph26c.height, " matches:", ph26c["match_key"].n_unique())
rif = ph26c.filter((pl.col("weapon_name").is_in(["ak47", "m4a1"])) & (pl.col("hit_box_code") == 1))
print("head hits AK47 (clean):", rif.filter(pl.col("weapon_name") == "ak47").height,
      " M4 (m4a1, clean):", rif.filter(pl.col("weapon_name") == "m4a1").height)
print("AK head hits matches:", rif.filter(pl.col("weapon_name") == "ak47")["match_key"].n_unique())
print("M4 head hits matches:", rif.filter(pl.col("weapon_name") == "m4a1")["match_key"].n_unique())
# mean health_removed by weapon for head
print(rif.group_by("weapon_name").agg(pl.len().alias("n"), pl.col("health_removed").mean().alias("mean_hr"), pl.col("health_removed").median().alias("med_hr"), (pl.col("health") == 0).mean().alias("share_health0")).to_dicts())
# analyst-style full-HP test (health + health_removed == 100)
rif2 = rif.with_columns((pl.col("health") + pl.col("health_removed")).alias("hp_before_a"))
print("analyst hp_before_a==100 counts by weapon:")
print(rif2.group_by("weapon_name").agg(pl.len().alias("n_all"), (pl.col("hp_before_a") == 100).sum().alias("n_full_a"), ((pl.col("hp_before_a") == 100) & (pl.col("health") == 0)).sum().alias("n_full_lethal_a")).to_dicts())
print("hits with health_removed > 100 (AK, M4):", rif.group_by("weapon_name").agg((pl.col("health_removed") > 100).sum().alias("gt100"), (pl.col("health_removed") >= 100).sum().alias("ge100")).to_dicts())
print("health_removed sample AK head (first 15):", rif.filter(pl.col("weapon_name") == "ak47").select(["health", "health_removed", "armor", "armor_removed", "tick"]).head(15).to_dicts())
# negative / zero health_removed checks
print("health_removed<=0 in head:", rif.filter(pl.col("health_removed") <= 0).height, " health<0:", rif.filter(pl.col("health") < 0).height)
print("effective_health_removed vs health_removed (head AK) mismatch count:", rif.filter(pl.col("weapon_name") == "ak47").filter(pl.col("effective_health_removed") != pl.col("health_removed")).height)
print("ALL DONE")
