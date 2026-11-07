"""verify_D2_3 probe: event types, side codes and round numbering in the 2024 and 2026 tomes.

Read-only. Reads small projections of the tome tables. Writes nothing.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def read(label, ch, cols):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def main():
    for label in ["2024", "2026"]:
        print(f"\n################ {label} ################")
        rs = read(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase", "is_warmup"])
        print("round_state rows:", rs.height)
        print(rs.group_by("event_type").len().sort("len", descending=True))
        bs = read(label, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
        print("bomb_state event types:")
        print(bs.group_by(["event_type", "player_team_code"]).len().sort(["event_type", "player_team_code"]))
        pd_ = read(label, "player_death", ["match_key", "round", "player_team_code"])
        print("player_death player_team_code counts:")
        print(pd_.group_by("player_team_code").len().sort("player_team_code"))
        re_ = read(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
        print("round_end rows:", re_.height, " matches:", re_["match_key"].n_unique())
        dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
        print("round_end duplicated (match, round) keys:", dup.height)
        print("round_end winner_team_code counts:")
        print(re_.group_by("winner_team_code").len().sort("winner_team_code"))
        print("win_reason_code by winner_team_code:")
        print(re_.group_by(["winner_team_code", "win_reason_code"]).len().sort(["winner_team_code", "win_reason_code"]))


if __name__ == "__main__":
    main()
