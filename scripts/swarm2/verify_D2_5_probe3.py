"""verify D2-5 probe 3: the single D2 row where D2's reason mapping (18 -> CT) disagrees with R (2).

Looks up the match key by prefix, prints round_end, freeze_end scores, bomb events and player_info side counts for rounds 9-11.
Read-only. Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_probe3.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
PREFIX = "csds/2024/02/02/m_partial"


def files(ch):
    fd = sorted(glob.glob(str(TOME / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


def read(ch, cols, where):
    fs = files(ch)
    have = pl.read_parquet_schema(fs[0]).keys()
    use = [c for c in cols if c in have]
    return pl.scan_parquet(fs).select(use).filter(where).collect()


mk_df = read("round_end", ["match_key"], pl.col("match_key").str.starts_with(PREFIX)).unique()
mks = mk_df["match_key"].to_list()
print("matching match keys:", mks)
MK = mks[0]

re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"],
           pl.col("match_key") == MK)
print("\nround_end (rounds 8-12):")
print(re_.filter(pl.col("round").is_between(8, 12)).sort("round"))

rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
          (pl.col("match_key") == MK) & pl.col("event_type").is_in(["round_freeze_end", "round_end", "round_officially_ended", "cs_win_panel_round"]))
print("\nround_state freeze_end / round_end / officially_ended / win_panel (rounds 8-12):")
print(rs.filter(pl.col("round").is_between(8, 12)).sort(["tick"]))

bs = read("bomb_state", ["match_key", "round", "tick", "event_type"], pl.col("match_key") == MK)
print("\nbomb_state (rounds 8-12):")
print(bs.filter(pl.col("round").is_between(8, 12)).sort("tick"))

pi = read("player_info", ["match_key", "round", "player_id", "team_code"], pl.col("match_key") == MK)
print("\nplayer_info team_code counts (rounds 8-12):")
print(pi.filter(pl.col("round").is_between(8, 12)).group_by(["round", "team_code"]).len().sort(["round", "team_code"]))
