"""P1: verify the item-code -> weapon-name mapping from basic_weapon_fire before using it.

Reads player_weapon_code and weapon_name (2 columns, lazy) from all 5 weapon_fire pages,
and the inv_secondary codes from the 1 Hz sample. Prints compact tables only.
"""
import glob
import os

import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample\players_1hz.parquet"

wf_folder = glob.glob(os.path.join(TOME, "basic_weapon_fire.*"))[0]
wf_files = sorted(glob.glob(os.path.join(wf_folder, "dataframe_*")))
wf = pl.scan_parquet(wf_files).select(["player_weapon_code", "weapon_name"])

pairs = (wf.group_by(["player_weapon_code", "weapon_name"]).agg(pl.len().alias("rows"))
         .collect())
total_rows = int(pairs["rows"].sum())
print("weapon_fire total rows (2-col scan):", total_rows)

# per code: total, top name, top share, number of distinct names
per_code = (pairs.sort(["player_weapon_code", "rows"], descending=[False, True])
            .group_by("player_weapon_code", maintain_order=True)
            .agg([
                pl.col("rows").sum().alias("code_rows"),
                pl.col("weapon_name").first().alias("top_name"),
                pl.col("rows").first().alias("top_rows"),
                pl.col("weapon_name").n_unique().alias("n_names"),
                pl.col("weapon_name").slice(1, 3).alias("other_names_top3"),
            ])
            .with_columns((pl.col("top_rows") / pl.col("code_rows")).round(5).alias("top_share"))
            .sort("code_rows", descending=True))

PISTOL_CANDIDATES = [0, 1, 2, 3, 4, 30, 32, 36, 61, 63, 64]
print("\n== Pistol-candidate codes (weapon_fire names) ==")
with pl.Config(tbl_rows=100, tbl_cols=12, tbl_width_chars=260, fmt_str_lengths=80):
    print(per_code.filter(pl.col("player_weapon_code").is_in(PISTOL_CANDIDATES)))

print("\n== Codes with more than one name (conflicts), top 25 by code rows ==")
with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=260, fmt_str_lengths=80):
    print(per_code.filter(pl.col("n_names") > 1).head(25))

print("\n== Codes 59, 45, 500 detail ==")
with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=260, fmt_str_lengths=80):
    print(pairs.filter(pl.col("player_weapon_code").is_in([59, 45, 500]))
          .sort(["player_weapon_code", "rows"], descending=[False, True]).head(40))

# inv_secondary codes in the 1 Hz sample, with the pistol name from weapon_fire where available
samp = pl.scan_parquet(SAMPLE).select(["inv_secondary", "team_code", "round"])
sec = (samp.group_by(["inv_secondary"]).agg(pl.len().alias("player_rows"))
       .sort("player_rows", descending=True).collect())
print("\n== inv_secondary codes in 1 Hz sample (all values) ==")
name_map = (per_code.select(["player_weapon_code", "top_name", "top_share", "n_names"]))
joined = sec.join(name_map, left_on="inv_secondary", right_on="player_weapon_code", how="left")
with pl.Config(tbl_rows=60, tbl_cols=8, tbl_width_chars=200, fmt_str_lengths=40):
    print(joined)

print("\n== team_code values in sample ==")
print(samp.group_by("team_code").agg(pl.len().alias("n")).collect())
print("\n== round range in sample ==")
r = samp.select(pl.col("round").min().alias("min"), pl.col("round").max().alias("max")).collect()
print(r)
