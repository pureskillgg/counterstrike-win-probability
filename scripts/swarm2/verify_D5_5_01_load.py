r"""Verifier for D5-5 (round-13 freeze-end score step vs round_end winner).

Loads ONLY the 2024 tome on G: (header, round_end, round_state, bomb_state) with
column and event-type projection, writes intermediate parquet to the session scratchpad
(not under E:, not in the project tree). Prints schema-level sanity checks.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_01_load.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(100)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
SCR.mkdir(parents=True, exist_ok=True)


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    assert fs, ch
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


H = (pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))[0])
     .select(["key", "match_id", "map_name", "match_date", "tick_rate",
              "t_starters_score_final", "ct_starters_score_final", "match_type"])
     .collect())
H = H.with_columns(day=pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1))
H.write_parquet(SCR / "header.parquet")
print("header rows", H.height, "unique key", H["key"].n_unique())

RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                 "win_reason_message", "legacy_code", "player_count"]).collect())
RE.write_parquet(SCR / "round_end.parquet")
print("round_end rows", RE.height, "pairs", RE.select(["match_key", "round"]).unique().height)

RS_TYPES = ["round_freeze_end", "round_start", "round_end", "bomb_planted", "bomb_defused", "bomb_exploded"]
RS = (scan("round_state").filter(pl.col("event_type").is_in(RS_TYPES))
      .select(["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw",
               "ct_score_raw", "phase", "is_warmup"]).collect())
RS.write_parquet(SCR / "round_state_sub.parquet")
print("round_state (subset) rows", RS.height)
print(RS.group_by("event_type").len().sort("len", descending=True))

BS = (scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect())
BS.write_parquet(SCR / "bomb_state_sub.parquet")
print("bomb_state (subset) rows", BS.height)
print(BS.group_by("event_type").len().sort("event_type"))

# raw vs derived score fields
print("\nt_score == t_score_raw among freeze_end rows:")
fe = RS.filter(pl.col("event_type") == "round_freeze_end")
print(fe.select((pl.col("t_score") == pl.col("t_score_raw")).mean().alias("t_eq_raw"),
                (pl.col("ct_score") == pl.col("ct_score_raw")).mean().alias("ct_eq_raw"),
                pl.len().alias("n")))
print(fe.filter((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw")))
      .head(10))
print("\nfreeze_end rows per (match,round) multiplicity:")
print(fe.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("pairs")).sort("n"))
print("round_end rows per (match,round) multiplicity:")
print(RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).group_by("n").agg(pl.len().alias("pairs")).sort("n"))
print("\nround_end winner_team_code counts:")
print(RE.group_by("winner_team_code").len().sort("winner_team_code"))
print("win_reason_code x message:")
print(RE.group_by(["win_reason_code", "win_reason_message"]).len().sort(["win_reason_code"]))
print("\nmatches in round_end:", RE["match_key"].n_unique(), " in header:", H.height,
      " header keys not in round_end:", H.join(RE.select(pl.col("match_key").alias("key")).unique(), on="key", how="anti").height)
