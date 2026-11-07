# -*- coding: utf-8 -*-
"""
verify_A1_2_semantics.py  (alternative-explanation checks for A1-2)

Tests whether the side codes used by the identity check (basic_player_info.team_code and
basic_round_end.winner_team_code) carry the game semantics (2 = T, 3 = CT), using channels that
do not depend on the starter/swap logic:
  (1) round_end win_reason vs winner side, across all 2026 matches and for the claimed matches
  (2) bomb planter side (engine player_team_code on plant events, and player_info team_code in that round)
  (3) player_info team_code vs engine spawn/death player_team_code for the same player and round
  (4) for the claimed matches: start-T group by identity vs the engine end-side scores
Read-only, lazy scans with column projection.
"""
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CLAIMED = [
    "csds/2026/10/01/7368f3ae-7353-4f68-8d04-7c97f22ea2b8/csds",
    "csds/2026/10/01/ec8a424b-067f-4209-a42c-36d5a65cd97f/csds",
    "csds/2026/10/02/5429f41b-7721-4232-9784-bd9a3e2045a5/csds",
    "csds/2026/10/02/5f014ad2-63f8-415d-9952-5984b2924721/csds",
    "csds/2026/10/02/83d3ede2-5226-4035-8f32-87a32f083665/csds",
    "csds/2026/10/02/b2e50666-9b0b-4dcd-b8cc-59703e6a3064/csds",
    "csds/2026/10/02/d69b442a-8c83-4700-8a25-a035082e0ba8/csds",
    "csds/2026/10/02/e60396ec-5ff6-4a18-a2c9-b7187b650485/csds",
    "csds/2026/10/03/2093742d-edc0-4041-961c-a261f4f26876/csds",
    "csds/2026/10/03/cbb071c2-83a9-4167-b4c1-6fed1e145da0/csds",
]


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


def scan(name, cols):
    return pl.scan_parquet(chan(name)).select(cols).collect()


claimed = set(CLAIMED)
re_ = scan("basic_round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message"])
print("== (1) round_end win_reason vs winner side (all matches) ==")
ct = re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).agg(pl.len().alias("n")).sort(["win_reason_code", "winner_team_code"])
print(ct)
cl = re_.filter(pl.col("match_key").is_in(list(claimed)))
print("claimed-10 round_end win_reason x winner:")
print(cl.group_by(["win_reason_message", "winner_team_code"]).agg(pl.len().alias("n")).sort(["win_reason_message", "winner_team_code"]))

print("\n== (2) bomb events: event types ==")
for name in ["basic_bomb_action", "basic_bomb_state", "basic_bomb_defuse"]:
    try:
        b = scan(name, ["match_key", "round", "event_type"]) if name != "basic_bomb_defuse" else scan(name, ["match_key", "round"])
        print(name, "rows", b.height)
        if "event_type" in b.columns:
            print("   event_type counts:", b.group_by("event_type").agg(pl.len().alias("n")).sort("event_type").rows())
    except Exception as e:
        print(name, "ERR", str(e)[:200])

ba_cols = pl.scan_parquet(chan("basic_bomb_action")).collect_schema().names()
print("bomb_action columns:", ba_cols)
ba = scan("basic_bomb_action", ["match_key", "round", "event_type", "player_id", "player_team_code"])
print("bomb_action event_type x player_team_code:", ba.group_by(["event_type", "player_team_code"]).agg(pl.len().alias("n")).sort(["event_type", "player_team_code"]).rows())

pi = scan("basic_player_info", ["match_key", "round", "player_id", "team_code"])
pp = scan("basic_player_personal", ["match_key", "round", "player_id", "steam_id"])
# steam mapping per match
pmap_df = pp.group_by(["match_key", "player_id"]).agg(pl.col("steam_id").first().alias("sid"))
pinfo = pi.join(pmap_df, on=["match_key", "player_id"], how="left")

# planter check: player_info team_code of the planter in that round, and engine player_team_code
plants = ba.filter(pl.col("event_type").str.contains("(?i)plant")).select(["match_key", "round", "player_id", "player_team_code"])
pl_pi = plants.join(pi, on=["match_key", "round", "player_id"], how="left")
print("\nplant events:", plants.height, "| player_info team_code of planter (value counts):",
      pl_pi.group_by("team_code").agg(pl.len().alias("n")).sort("team_code").rows())
print("plant events engine player_team_code:", plants.group_by("player_team_code").agg(pl.len().alias("n")).sort("player_team_code").rows())
cp = pl_pi.filter(pl.col("match_key").is_in(list(claimed)))
print("claimed-10 plant events:", cp.height, "| planter player_info team_code:", cp.group_by("team_code").agg(pl.len().alias("n")).sort("team_code").rows())
print("claimed-10 plant engine player_team_code:", plants.filter(pl.col("match_key").is_in(list(claimed))).group_by("player_team_code").agg(pl.len().alias("n")).sort("player_team_code").rows())

print("\n== (3) player_info team_code vs engine spawn/death player_team_code (same player, round) ==")
sp = scan("basic_player_spawn", ["match_key", "round", "player_id", "player_team_code"])
dt = scan("basic_player_death", ["match_key", "round", "player_id", "player_team_code"])
ev = pl.concat([sp, dt], how="vertical").unique()
cmp = ev.join(pi, on=["match_key", "round", "player_id"], how="inner")
print("event rows joined to player_info:", cmp.height)
print("agree (player_team_code == team_code):", cmp.filter(pl.col("player_team_code") == pl.col("team_code")).height,
      "| disagree:", cmp.filter(pl.col("player_team_code") != pl.col("team_code")).height)
cmp_c = cmp.filter(pl.col("match_key").is_in(list(claimed)))
print("claimed-10 agree:", cmp_c.filter(pl.col("player_team_code") == pl.col("team_code")).height,
      "| disagree:", cmp_c.filter(pl.col("player_team_code") != pl.col("team_code")).height)
dis = cmp.filter(pl.col("player_team_code") != pl.col("team_code"))
print("disagree by match (top 15):", dis.group_by("match_key").agg(pl.len().alias("n")).sort("n", descending=True).head(15).rows())
