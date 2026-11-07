r"""D5-8 verifier: raw tick ordering dump for a few regime-A mismatch matches (rounds 11..15).

Shows, per round: round_end channel (tick, winner, reason), round_state freeze_end/round_start/round_end ticks,
and bomb_state events (tick, type, team). Used to see whether round-13 bomb events fall inside round 13's own window.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_tickdump.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(70)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")


def scan(ch):
    fd = os.path.join(BASE, f"basic_{ch}{SUF}")
    return pl.scan_parquet(sorted(glob.glob(os.path.join(fd, "dataframe_*"))), missing_columns="insert",
                           extra_columns="ignore")


R = pl.read_parquet(OUT / "rounds_v2.parquet")
mm = R.filter((pl.col("grp") == "lateA") & (pl.col("round") == 13) & pl.col("bside").is_not_null()
              & (pl.col("winner") != pl.col("bside")))
keys = mm.select("match_key").head(3)["match_key"].to_list()
print("sample keys:", [k.split("/")[-2][:8] for k in keys])
for k in keys:
    print("\n=== match", k.split("/")[-2][:8], "day", k.split("/")[3])
    rr = R.filter(pl.col("match_key") == k).filter(pl.col("round").is_between(10, 16)).select(
        ["round", "end_tick", "fe_tick", "fe_t", "fe_ct", "fe_phase", "winner", "reason", "bside", "bomb_exp", "bomb_def",
         "planter"]).sort("round")
    print(rr)
    rs = scan("round_state").filter(pl.col("match_key") == k).filter(pl.col("round").is_between(10, 16)).select(
        ["round", "tick", "event_type", "phase"]).filter(pl.col("event_type").is_in(
        ["round_start", "round_freeze_end", "round_end", "round_officially_ended", "cs_win_panel_round", "bomb_planted",
         "bomb_defused", "bomb_exploded"])).sort(["tick"]).collect()
    print("round_state events (tick-ordered):")
    print(rs)
    bs = scan("bomb_state").filter(pl.col("match_key") == k).filter(pl.col("round").is_between(10, 16)).select(
        ["round", "tick", "event_type", "player_team_code"]).sort("tick").collect()
    print("bomb_state events (tick-ordered):")
    print(bs)
    re_ = scan("round_end").filter(pl.col("match_key") == k).filter(pl.col("round").is_between(10, 16)).select(
        ["round", "tick", "winner_team_code", "win_reason_code"]).sort("tick").collect()
    print("round_end channel (tick-ordered):")
    print(re_)
