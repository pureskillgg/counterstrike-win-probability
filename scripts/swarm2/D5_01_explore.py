"""D5 step 1: vocabulary and shape of round channels in the 2024 tome.

Prints event_type / phase / is_warmup counts from round_state, reason-code and
legacy_code counts from round_end, round_start row multiplicity, and header map counts.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D5_01_explore.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from D5_common import header, read  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(120)

H = header()
print("header rows:", H.height, "unique key:", H["key"].n_unique(), "unique match_id:", H["match_id"].n_unique())
print("map_name counts:")
print(H.group_by("map_name").len().sort("len", descending=True))
print("match_type counts:")
print(H.group_by("match_type").len())
print("tick_rate counts:")
print(H.group_by("tick_rate").len())

RS = read("round_state", ["match_key", "round", "tick", "event_type", "phase", "is_warmup", "t_score", "ct_score"])
print("\nround_state rows:", RS.height)
print("event_type counts:")
print(RS.group_by("event_type").len().sort("len", descending=True))
print("phase counts:")
print(RS.group_by("phase").len().sort("len", descending=True))
print("is_warmup counts:", RS.group_by("is_warmup").len().to_dicts())

RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                        "win_reason_message", "legacy_code", "player_count"])
print("\nround_end rows:", RE.height, "unique (match,round):", RE.select(["match_key", "round"]).unique().height)
print("win_reason_code x win_reason_message:")
print(RE.group_by(["win_reason_code", "win_reason_message"]).len().sort(["win_reason_code", "win_reason_message"]))
print("legacy_code counts:")
print(RE.group_by("legacy_code").len().sort("legacy_code"))
print("player_count counts:")
print(RE.group_by("player_count").len().sort("player_count"))
print("winner_team_code counts:")
print(RE.group_by("winner_team_code").len().sort("winner_team_code"))

RST = read("round_start", ["match_key", "round", "tick"])
print("\nround_start rows:", RST.height, "unique (match,round):", RST.select(["match_key", "round"]).unique().height)
print("round_start rows per (match,round) multiplicity:")
print(RST.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("pairs")).sort("n"))

print("\nround_end rows per (match,round) multiplicity:")
print(RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("pairs")).sort("n"))

print("\nround value range in round_end:", RE["round"].min(), RE["round"].max())
print("round_state round value range:", RS["round"].min(), RS["round"].max())
print("round_start round value range:", RST["round"].min(), RST["round"].max())
print("round_end distinct rounds per match, value counts:")
per_m = RE.group_by("match_key").agg(pl.col("round").n_unique().alias("n_rounds"), pl.len().alias("n_rows"))
print(per_m.group_by("n_rows").agg(pl.len().alias("matches")).sort("n_rows").head(40))

print("\nmatch_key sample (round_end) vs header key sample:")
print(RE["match_key"].head(2).to_list(), H["key"].head(2).to_list())
print("round_end match_keys not in header:", RE.select("match_key").unique().join(H.select(pl.col("key").alias("match_key")), on="match_key", how="anti").height)
