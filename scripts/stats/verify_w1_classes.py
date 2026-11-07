"""Cross-tab death classification inputs to explain differences vs the W1 report."""
import glob
import os

import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def files(ch):
    return sorted(glob.glob(os.path.join(TOME, f"{ch}.2026-09-18,2026-10-08", "dataframe_*")))


d = pl.scan_parquet(files("basic_player_death")).select(
    [
        "match_key",
        "player_id",
        "attacker_id",
        "assister_id",
        "player_id_fixed",
        "attacker_id_fixed",
        "player_team_code",
        "attacker_team_code",
        "weapon_name",
        "player_weapon_code",
        "attacker_weapon_code",
    ]
).collect()

print("rows", d.height)
print("player_id_fixed null:", d["player_id_fixed"].null_count())
print("attacker_id_fixed null:", d["attacker_id_fixed"].null_count())
print("attacker_id==player_id:", int((d["attacker_id"] == d["player_id"]).sum()))
print("attacker_id_fixed==player_id_fixed:", int((d["attacker_id_fixed"] == d["player_id_fixed"]).sum()))
print("attacker_id null:", d["attacker_id"].null_count())
print("attacker_id == 0:", int((d["attacker_id"] == 0).sum()))
print("attacker_id_fixed == 0:", int((d["attacker_id_fixed"] == 0).sum()))

print("\nCrosstab attacker_team_code (null shown) x player_team_code")
ct = d.group_by(["attacker_team_code", "player_team_code"]).agg(pl.len().alias("n")).sort(["attacker_team_code", "player_team_code"], nulls_last=True)
print(ct)

print("\nSelf-candidates (attacker_id==player_id) by team codes and weapon")
selfs = d.filter(pl.col("attacker_id") == pl.col("player_id"))
print(selfs.group_by(["attacker_team_code", "player_team_code"]).agg(pl.len().alias("n")))
print(selfs["weapon_name"].value_counts(sort=True).head(12))
print("self with attacker_id==0:", int((selfs["attacker_id"] == 0).sum()))

print("\nattacker_id == 0 rows: team codes and weapon")
z = d.filter(pl.col("attacker_id") == 0)
print(z.group_by(["attacker_team_code", "player_team_code"]).agg(pl.len().alias("n")))
print(z["weapon_name"].value_counts(sort=True).head(12))

print("\nattacker_id null rows: team codes and weapon")
nn = d.filter(pl.col("attacker_id").is_null())
print(nn.group_by(["attacker_team_code", "player_team_code"]).agg(pl.len().alias("n")))
print(nn["weapon_name"].value_counts(sort=True).head(12))

print("\nRows with attacker_team_code null: weapon and attacker_id")
an = d.filter(pl.col("attacker_team_code").is_null())
print(an.group_by(["attacker_id", "weapon_name"]).agg(pl.len().alias("n")).sort("n", descending=True).head(15))

print("\nRows with player_team_code null: attacker_team_code distribution")
pn = d.filter(pl.col("player_team_code").is_null())
print(pn.group_by("attacker_team_code").agg(pl.len().alias("n")))
print(pn["weapon_name"].value_counts(sort=True).head(8))
