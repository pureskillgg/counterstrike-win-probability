import glob, os
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"

def pages(channel):
    d = os.path.join(BASE, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

ph = pl.concat([pl.read_parquet(p) for p in pages("player_hurt")], how="diagonal_relaxed")
print("player_hurt rows:", ph.height)
print("match_key prefixes:", ph.select(pl.col("match_key").str.slice(0, 9).alias("pfx")).group_by("pfx").len().to_dicts())
print("distinct matches:", ph["match_key"].n_unique())
ph26 = ph.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("2026 rows:", ph26.height, "matches:", ph26["match_key"].n_unique())
print("weapon_name values (2026):")
print(ph26.group_by("weapon_name").len().sort("len", descending=True).to_dicts())
print("hit_box_code (2026):")
print(ph26.group_by("hit_box_code").len().sort("hit_box_code").to_dicts())
rifle = ph26.filter(pl.col("weapon_name").is_in(["ak47", "m4a1", "m4a4", "m4a1_silencer", "ak-47", "AK-47", "M4A1", "M4A4", "m4a1_silencer_off", "m4a1_silencer_on"]))
print("mean health_removed by hit_box_code (rifle hits, weapon_name matched list):")
print(rifle.group_by("hit_box_code").agg(pl.len().alias("n"), pl.col("health_removed").mean().alias("mean_hr")).sort("hit_box_code").to_dicts())
