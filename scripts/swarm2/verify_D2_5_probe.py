"""verify D2-5 probe: value distributions for round_end reason fields, bomb events, round_state events.

Read-only over tomes\\2024-02-g and tomes\\2026-10-01_to_08. Writes nothing except stdout.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


for label in ["2024", "2026"]:
    re_ = read(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                    "win_reason_message", "legacy_code", "player_count"])
    print(f"\n===== {label} round_end rows={re_.height} matches={re_['match_key'].n_unique()}")
    print("player_count values:", re_.group_by("player_count").len().sort("player_count").rows())
    print("dup (match,round) rows:", re_.height - re_.unique(["match_key", "round"]).height)
    print("winner_team_code values:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
    print("\nwin_reason_code x win_reason_message x winner_team_code (counts):")
    print(re_.group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len()
          .sort(["win_reason_code", "winner_team_code"]))
    print("\nlegacy_code x winner_team_code:")
    print(re_.group_by(["legacy_code", "winner_team_code"]).len().sort(["legacy_code", "winner_team_code"]))

    bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    print(f"\nbomb_state rows={bs.height}; event_type counts:")
    print(bs.group_by("event_type").len().sort("event_type"))

    rs = read(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                                     "t_score_raw", "ct_score_raw"])
    print(f"\nround_state rows={rs.height}; event_type counts:")
    print(rs.group_by("event_type").len().sort("event_type"))
