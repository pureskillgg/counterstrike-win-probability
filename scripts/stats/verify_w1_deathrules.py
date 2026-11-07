"""One attempt to reproduce the report's death-class totals (enemy 247,629; world 5,100; self 73; team 669; unknown 9)
by filling missing team codes from player_info (match_key, round, player_id) and testing a world-weapon rule."""
import glob
import os

import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"


def files(ch):
    return sorted(glob.glob(os.path.join(TOME, f"{ch}.2026-09-18,2026-10-08", "dataframe_*")))


d = pl.scan_parquet(files("basic_player_death")).select(
    ["match_key", "round", "player_id", "attacker_id", "player_team_code", "attacker_team_code", "weapon_name"]
).collect()

pi = (
    pl.scan_parquet(files("basic_player_info"))
    .select(["match_key", "round", "player_id", "team_code"])
    .filter(pl.col("team_code").is_in([2, 3]))
    .group_by(["match_key", "round", "player_id"])
    .agg(pl.col("team_code").first().alias("pi_team"))
    .collect()
)
print("player_info (match, round, player) team rows:", pi.height)

d = d.join(pi.rename({"player_id": "player_id"}), left_on=["match_key", "round", "player_id"],
           right_on=["match_key", "round", "player_id"], how="left")
d = d.join(
    pi.rename({"player_id": "attacker_id", "pi_team": "att_pi_team"}),
    left_on=["match_key", "round", "attacker_id"], right_on=["match_key", "round", "attacker_id"], how="left",
)
d = d.with_columns(
    pl.coalesce([pl.col("player_team_code"), pl.col("pi_team")]).alias("vic_team"),
    pl.coalesce([pl.col("attacker_team_code"), pl.col("att_pi_team")]).alias("atk_team"),
)
print("victim team still null:", d.filter(pl.col("vic_team").is_null()).height)
print("attacker team null after fill:", d.filter(pl.col("atk_team").is_null()).height)

self_ = (pl.col("attacker_id") == pl.col("player_id")) & pl.col("attacker_id").is_not_null()
for label, world_rule in [
    ("world = attacker team not in {2,3}", ~pl.col("atk_team").is_in([2, 3])),
    ("world = weapon 'world' or attacker team not in {2,3}",
     (pl.col("weapon_name") == "world") | ~pl.col("atk_team").is_in([2, 3])),
    ("world = weapon 'world' or (attacker team null and victim team known) or attacker_team 0",
     (pl.col("weapon_name") == "world")
     | (pl.col("atk_team").is_null() & pl.col("vic_team").is_not_null())
     | (pl.col("atk_team") == 0)),
]:
    cls = (
        pl.when(self_ & (pl.col("weapon_name") != "world")).then(pl.lit("self"))
        .when(world_rule).then(pl.lit("world"))
        .when(pl.col("atk_team") == pl.col("vic_team")).then(pl.lit("team"))
        .when(pl.col("atk_team").is_in([2, 3]) & pl.col("vic_team").is_in([2, 3])).then(pl.lit("enemy"))
        .otherwise(pl.lit("unknown"))
    )
    cc = d.select(cls.alias("c"))["c"].value_counts(sort=True).rows()
    print(label, "->", dict(cc))

# which rows the report would have to move: self-world (self rule) and the rest
print("self with weapon world:", d.filter(self_ & (pl.col("weapon_name") == "world")).height)
print("weapon world total:", d.filter(pl.col("weapon_name") == "world").height)
