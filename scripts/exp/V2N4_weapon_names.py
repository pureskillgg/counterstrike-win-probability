import glob, os
import polars as pl
BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
def pages(ch):
    return sorted(glob.glob(os.path.join(BASE, f"basic_{ch}.COMBINED", "dataframe_*")))
pdth = pl.concat([pl.read_parquet(p) for p in pages("player_death")], how="diagonal_relaxed")
pdth = pdth.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("death 2026 weapon names with m4/ak:", pdth.filter(pl.col("weapon_name").str.contains("m4|ak")).group_by("weapon_name").len().sort("weapon_name").to_dicts())
ph = pl.concat([pl.read_parquet(p) for p in pages("player_hurt")], how="diagonal_relaxed")
ph = ph.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print("hurt 2026 weapon names with m4/ak:", ph.filter(pl.col("weapon_name").str.contains("m4|ak")).group_by("weapon_name").len().sort("weapon_name").to_dicts())
print("death 2026 headshot kills by weapon (ak/m4 names):", pdth.filter(pl.col("weapon_name").str.contains("m4|ak")).group_by(["weapon_name", "is_headshot"]).len().sort(["weapon_name", "is_headshot"]).to_dicts())
print("ALL DONE")
