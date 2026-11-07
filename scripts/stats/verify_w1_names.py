"""List every raw weapon_name in enemy kills and shots, to test the silencer-off merge used in the W1 report."""
import glob
import os

import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def files(ch):
    return sorted(glob.glob(os.path.join(TOME, f"{ch}.2026-09-18,2026-10-08", "dataframe_*")))


shots = (
    pl.scan_parquet(files("basic_weapon_fire")).select(["weapon_name"])
    .group_by("weapon_name").agg(pl.len().alias("shots")).collect()
)
kills = (
    pl.scan_parquet(files("basic_player_death")).select(["weapon_name"])
    .group_by("weapon_name").agg(pl.len().alias("deaths_any_side")).collect()
)
both = shots.join(kills, on="weapon_name", how="full", coalesce=True).sort("shots", descending=True, nulls_last=True)
print(both.with_columns(pl.col("shots").fill_null(0), pl.col("deaths_any_side").fill_null(0)).to_pandas().to_string()
      if False else both.to_init_repr()[:0] or both)
print("names containing 'off' or 'silencer':",
      [n for n in both["weapon_name"].to_list() if n and ("off" in n or "silencer" in n)])
