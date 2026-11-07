"""verify D2-6: probe join keys, event vocabularies, duplicate structure (projection only)."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def scan(label, ch):
    if ch == "header":
        d = TOME[label] / "header"
    else:
        d = Path(sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0])
    return pl.scan_parquet(sorted(glob.glob(str(d / "dataframe_*"))))


for label in ["2024", "2026"]:
    print(f"==== {label} ====")
    h = scan(label, "header").select(["key", "match_id", "match_date", "map_name",
                                       "t_starters_score_final", "ct_starters_score_final"]).collect()
    print("header rows", h.height, "unique key", h["key"].n_unique(), "unique match_id", h["match_id"].n_unique())
    print(h.head(3))
    re_ = scan(label, "round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                                            "player_count"]).collect()
    print("round_end rows", re_.height, "unique (mk,round)", re_.select(["match_key", "round"]).unique().height)
    print("round_end match_key sample:", re_["match_key"].head(3).to_list())
    print("winner_team_code counts:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
    print("player_count counts:", re_.group_by("player_count").len().sort("player_count").rows())
    rs = scan(label, "round_state").select(["match_key", "round", "tick", "event_type"]).collect()
    print("round_state event types:", rs.group_by("event_type").len().sort("len", descending=True).rows()[:30])
    bs = scan(label, "bomb_state").select(["match_key", "round", "tick", "event_type"]).collect()
    print("bomb_state event types:", bs.group_by("event_type").len().sort("len", descending=True).rows()[:30])
    pi = scan(label, "player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
    print("player_info rows", pi.height, "unique (mk,round,pid)", pi.select(["match_key", "round", "player_id"]).unique().height,
          "team_code counts", pi.group_by("team_code").len().sort("team_code").rows())
    keys_rs = set(rs["match_key"].unique().to_list())
    keys_re = set(re_["match_key"].unique().to_list())
    hk = set(h["key"].to_list())
    print("match_key count in round_end:", len(keys_re), " in round_state:", len(keys_rs),
          " matches in header:", len(hk))
    print("round_end keys not in header key:", len(keys_re - hk), " header keys not in round_end:", len(hk - keys_re))
    print("example round_end keys not in header:", list(keys_re - hk)[:2])
    print("example header keys not in round_end:", list(hk - keys_re)[:2])
