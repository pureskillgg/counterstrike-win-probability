r"""Verifier for D5-5, stage 7: raw source check for the worked-example match (read-only on E:).

Compares the raw per-match parquet files on E: (tail-recovered 2024 copy) with the tome rows for the
same match: round_end (winner, reason), round_state freeze_end scores, bomb_state events.
Tells whether the round-13 winner inversion is already present in the raw source or introduced in the tome.
Writes nothing under E:. Reads only this one match.

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_08_raw.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12"
           r"\05d51edb-04bc-420c-91b9-4bb95317ca19")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
MK = "csds/2024/02/12/05d51edb-04bc-420c-91b9-4bb95317ca19/csds"

re_raw = pl.read_parquet(RAW / "round_end")
print("RAW round_end columns:", re_raw.columns)
print(re_raw.select([c for c in ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                                 "player_count"] if c in re_raw.columns]).filter(pl.col("round").is_between(11, 15)).sort("round"))

rs_raw = pl.read_parquet(RAW / "round_state")
print("\nRAW round_state columns:", rs_raw.columns)
fe_raw = rs_raw.filter(pl.col("event_type") == "round_freeze_end").select(
    ["round", "tick", "t_score", "ct_score"]).sort(["round", "tick"])
print("RAW freeze_end (11..15):")
print(fe_raw.filter(pl.col("round").is_between(11, 15)))

bs_raw = pl.read_parquet(RAW / "bomb_state")
print("\nRAW bomb_state (bomb events, rounds 12..14):")
print(bs_raw.filter(pl.col("event_type").is_in(["bomb_planted", "bomb_exploded", "bomb_defused"]) &
                    pl.col("round").is_between(12, 14)).select(["round", "tick", "event_type", "player_team_code"]).sort(["round", "tick"]))

# tome rows for the same match (from scratch extracts written by stage 1)
RE_t = pl.read_parquet(SCR / "round_end.parquet").filter((pl.col("match_key") == MK) & pl.col("round").is_between(11, 15))
RS_t = pl.read_parquet(SCR / "round_state_sub.parquet").filter(
    (pl.col("match_key") == MK) & (pl.col("event_type") == "round_freeze_end") & pl.col("round").is_between(11, 15))
BS_t = pl.read_parquet(SCR / "bomb_state_sub.parquet").filter(
    (pl.col("match_key") == MK) & pl.col("round").is_between(12, 14))
print("\nTOME round_end (11..15):")
print(RE_t.select(["round", "tick", "winner_team_code", "win_reason_code"]).sort("round"))
print("TOME freeze_end (11..15):")
print(RS_t.select(["round", "tick", "t_score", "ct_score"]).sort("round"))
print("TOME bomb rows (12..14):")
print(BS_t.select(["round", "tick", "event_type", "player_team_code"]).sort(["round", "tick"]))

# row-level comparison of round_end (raw vs tome) for the whole match
re_all_raw = re_raw.select(["round", "tick", "winner_team_code", "win_reason_code"]).sort(["round", "tick"])
re_all_t = pl.read_parquet(SCR / "round_end.parquet").filter(pl.col("match_key") == MK).select(
    ["round", "tick", "winner_team_code", "win_reason_code"]).sort(["round", "tick"])
cmp = re_all_raw.join(re_all_t, on=["round", "tick"], how="full", suffix="_tome", coalesce=True)
print("\nround_end raw vs tome (full outer join on round,tick): rows", cmp.height,
      " mismatched winner:", cmp.filter(pl.col("winner_team_code") != pl.col("winner_team_code_tome")).height,
      " mismatched reason:", cmp.filter(pl.col("win_reason_code") != pl.col("win_reason_code_tome")).height)
print(cmp.filter(pl.col("round").is_between(11, 15)))
