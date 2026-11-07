"""Roster check for null-coded deaths. Do null-coded victim ids belong to the match roster (coded players)?
Prints every death in the flagged example rounds (including null-coded), with the roster side of each player id,
and the id distribution of null-coded deaths vs coded deaths in full rounds.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(14)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
import glob as _g


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(_g.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


de = read("player_death", ["match_key", "round", "tick", "player_id", "attacker_id", "player_team_code",
                           "attacker_team_code", "weapon_name"])
rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
roster = de.filter(pl.col("player_team_code").is_in([2, 3])).group_by(["match_key", "player_id"]).agg(
    pl.col("player_team_code").n_unique().alias("nside"), pl.col("player_team_code").mode().first().alias("side_any"),
    pl.len().alias("n_deaths_coded"))
print("player_id ranges: coded victims min/max", de.filter(pl.col("player_team_code").is_in([2, 3]))["player_id"].min(),
      de.filter(pl.col("player_team_code").is_in([2, 3]))["player_id"].max())
nul = de.filter(pl.col("player_team_code").is_null())
print("null-coded victim id quantiles:", nul["player_id"].quantile(0.1), nul["player_id"].quantile(0.5), nul["player_id"].quantile(0.9))
print("null-coded victim: id >= 65000 share:", nul.filter(pl.col("player_id") >= 65000).height / nul.height)
print("null-coded victim: self-kill (attacker == victim) share:", nul.filter(pl.col("attacker_id") == pl.col("player_id")).height / nul.height)
print("null-coded victim: victim id in roster (any coded death in same match):",
      nul.join(roster.select(["match_key", "player_id"]), on=["match_key", "player_id"], how="inner").height, "of", nul.height)

for mk, rnd in [("csds/2024/02/19/4634b4d3-7955-41e4-865e-b7858da38a94/csds", 13),
                ("csds/2024/02/24/bd6e8712-5267-4816-90ad-da2d02c62878/csds", 13)]:
    print("\n=== match", mk, "round", rnd)
    ends = rs.filter((pl.col("match_key") == mk) & (pl.col("round") == rnd) & pl.col("event_type").is_in(
        ["round_freeze_end", "round_officially_ended", "bomb_planted"])).sort("tick")
    print(ends.select(["tick", "event_type"]))
    d = de.filter((pl.col("match_key") == mk) & (pl.col("round") == rnd)).sort("tick")
    d = d.join(roster.filter(pl.col("match_key") == mk).select(["player_id", "side_any"]), on="player_id", how="left")
    print(d.select(["tick", "player_id", "attacker_id", "player_team_code", "attacker_team_code", "side_any", "weapon_name"]))
