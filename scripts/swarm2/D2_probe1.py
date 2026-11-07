"""D2 probe 1: row structure of round_end, round_start, player_info, bomb_state, round_state in 2024 and 2026."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(80)
ROOT = Path(r"G:\pureskill\csds-cs2")
T = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds", "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def scan(label, ch):
    fd = sorted(glob.glob(str(T[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*")))


for label in ["2024", "2026"]:
    re_ = scan(label, "round_end").collect()
    print(f"\n===== {label} round_end rows={re_.height} matches={re_['match_key'].n_unique()}")
    print("player_count dist:", re_.group_by("player_count").len().sort("player_count").rows())
    print("winner dist:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
    print("win_reason dist:", re_.group_by(["win_reason_code", "win_reason_message"]).len().sort("len", descending=True).head(20).rows())
    print("sample match_key:", re_["match_key"][0])
    dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    print("(match,round) with >1 round_end row:", dup.height)
    print(re_.head(6))
    rs = scan(label, "round_start").collect()
    print(f"round_start rows={rs.height} distinct (match,round)={rs.select(['match_key','round']).unique().height}")
    bs = scan(label, "bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
    print("bomb_state event types:", bs.group_by("event_type").len().sort("len", descending=True).rows())
    print("bomb_state planter team codes by event:", bs.group_by(["event_type", "player_team_code"]).len().sort(["event_type", "player_team_code"]).rows())
    pi = scan(label, "player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
    print(f"player_info rows={pi.height}; per (match,round) distinct players median-ish:")
    per = pi.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("np"))
    print(per["np"].value_counts().sort("np").rows()[:15])
    print("player_info team_code dist:", pi.group_by("team_code").len().sort("team_code").rows())
    print("player_info round min/max:", pi["round"].min(), pi["round"].max())
    print("player_info sample rows:")
    print(pi.filter(pl.col("match_key") == pi["match_key"][0]).sort(["round", "player_id"]).head(12))
