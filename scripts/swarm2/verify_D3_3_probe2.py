"""verify D3-3 probe 2: event vocabularies, round_end reason crosstab, player_info columns (2024 G: tome). Read-only."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


rs = scan("round_state").select(["match_key", "round", "event_type"]).group_by("event_type").len().collect()
print("round_state event types:")
print(rs.sort("len", descending=True))

for ch in ["bomb_action", "bomb_state", "bomb_defuse"]:
    d = scan(ch).group_by("event_type").len().collect()
    print(f"\n{ch} event types:")
    print(d.sort("len", descending=True))

re_ = scan("round_end").select(["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message"]).collect()
print("\nround_end rows:", re_.height, " distinct (match,round):", re_.select(["match_key", "round"]).unique().height)
print("\nround_end winner x reason_code x message:")
print(re_.group_by(["winner_team_code", "win_reason_code", "win_reason_message"]).len().sort(["win_reason_code", "winner_team_code"]))

pi_cols = scan("player_info").collect_schema()
print("\nplayer_info schema:", pi_cols)
print(scan("player_info").head(3).collect())

ps_cols = scan("player_spawn").collect_schema()
print("\nplayer_spawn schema:", ps_cols)
