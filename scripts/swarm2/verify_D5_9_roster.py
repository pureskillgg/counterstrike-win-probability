"""verify D5-9 roster: distinct players per match from player_death (victims and attackers), by regime and format.

Purpose: round_end player_count is 20 for every regime-A match and 8 or 6 for every regime-B MR8 match.
If it does not measure mode, the regime split and the mode split are not separable from it. This script
counts the distinct player_ids that appear in kills for each match (a lower bound on roster size).

Reads only the 2024 tome (player_death channel, projected). Writes only to scratchpad (C:) and the log.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_9_roster.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\VD59")

fs = sorted(glob.glob(str(BASE / f"basic_player_death{SUF}" / "dataframe_*")))
DP = pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed").select(
    ["match_key", "round", "player_id", "player_team_code", "attacker_id", "attacker_team_code"]).collect()
print("player_death rows:", DP.height)
print("player_id null share:", DP.select(pl.col("player_id").is_null().mean()).item())

vict = DP.select(["match_key", pl.col("player_id").alias("pid"), pl.col("player_team_code").alias("team")])
att = DP.select(["match_key", pl.col("attacker_id").alias("pid"), pl.col("attacker_team_code").alias("team")]).drop_nulls()
ROST = pl.concat([vict, att]).drop_nulls().unique().group_by("match_key").agg(
    pl.col("pid").n_unique().alias("n_players"),
    pl.col("team").filter(pl.col("team").is_in([2, 3])).n_unique().alias("n_team_codes"))
ROST.write_parquet(SCR / "roster.parquet")

# per team: distinct players per (match, team code) as a rough per-side roster
PT = pl.concat([vict, att]).drop_nulls().filter(pl.col("team").is_in([2, 3])).unique().group_by(
    ["match_key", "team"]).agg(pl.col("pid").n_unique().alias("n_side"))
PTm = PT.group_by("match_key").agg(pl.col("n_side").max().alias("max_side"), pl.col("n_side").min().alias("min_side"))
ROST = ROST.join(PTm, on="match_key", how="left")

# join with regime and format (from the build tables)
MK = pl.read_parquet(SCR / "marks.parquet")
regA = set(MK.filter(pl.col("event_type") == "round_announce_match_start")["match_key"].to_list())
ann = {r["match_key"]: r["min_round"] for r in
       MK.filter(pl.col("event_type") == "round_announce_last_round_half").iter_rows(named=True)}
RE = pl.read_parquet(SCR / "end_raw.parquet")
pc_mode = {}
for mk, grp in RE.group_by("match_key"):
    c = Counter(grp["player_count"].to_list())
    pc_mode[mk[0]] = c.most_common(1)[0][0]

ROST = ROST.with_columns(
    regime=pl.when(pl.col("match_key").is_in(list(regA))).then(pl.lit("A")).otherwise(pl.lit("B")),
    ann=pl.col("match_key").replace_strict(ann, default=None, return_dtype=pl.Int64),
    pc=pl.col("match_key").replace_strict(pc_mode, default=None, return_dtype=pl.Int64))

print("\n=== distinct players per match (n_players), by regime and halftime label")
for (reg, a), g_ in ROST.group_by(["regime", "ann"]):
    dist = Counter(g_["n_players"].to_list())
    print(f"  regime {reg} ann {a}: matches {g_.height}, n_players top {dist.most_common(6)}")

print("\n=== distinct players per match vs round_end player_count (cross-tab, top rows)")
ct = Counter(zip(ROST["regime"].to_list(), ROST["pc"].to_list(), ROST["n_players"].to_list()))
for k, v in sorted(ct.items(), key=lambda kv: (kv[0][0], -kv[1]))[:40]:
    print("  ", k, v)

print("\n=== max distinct players on one side (team code), by regime and label")
for (reg, a), g_ in ROST.group_by(["regime", "ann"]):
    print(f"  regime {reg} ann {a}: max_side dist {Counter(g_['max_side'].to_list()).most_common(6)}")
