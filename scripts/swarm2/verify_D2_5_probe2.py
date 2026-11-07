"""verify D2-5 probe 2: one defect match, round by round, to see score / bomb / reason / team-code alignment.

Read-only. Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_probe2.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(300)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
MK = "csds/2024/02/13/90350d19-5452-4c58-bc8b-f780e796e026/csds"


def read(ch, cols):
    fd = sorted(glob.glob(str(TOME / f"basic_{ch}.*")))[0]
    lf = pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*")))
    return lf.filter(pl.col("match_key") == MK).select(cols).collect()


re_ = read("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"])
print("round_end for match:")
print(re_.sort("round"))

rs = read("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
print("\nround_state freeze_end and round_end rows:")
print(rs.filter(pl.col("event_type").is_in(["round_freeze_end", "round_end", "round_start"]))
      .sort(["tick"]).select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]))

bs = read("bomb_state", ["round", "tick", "event_type", "player_team_code"])
print("\nbomb_state events:")
print(bs.filter(pl.col("event_type") != "bomb_planted").sort("tick"))
print("bomb_planted rounds:", bs.filter(pl.col("event_type") == "bomb_planted").group_by("round").len().sort("round").rows())

pi = read("player_info", ["round", "player_id", "team_code"])
print("\nplayer_info team_code counts per round (code: n):")
g = pi.group_by(["round", "team_code"]).len().sort(["round", "team_code"])
print(g.group_by("round").agg(pl.col("team_code").alias("codes"), pl.col("len").alias("ns")).sort("round"))
