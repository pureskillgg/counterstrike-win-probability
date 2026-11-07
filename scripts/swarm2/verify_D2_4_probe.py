"""Verifier D2-4: structure probe (read-only). Counts, duplicates, player_count values, raw vs non-raw score fields."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


def rd(label, ch, cols, where=None):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    lf = pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*")))
    if where is not None:
        lf = lf.filter(where)
    return lf.select(cols).collect()


for label in ["2024", "2026"]:
    print(f"\n######## {label}")
    re_ = rd(label, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    print("round_end rows", re_.height, "matches", re_["match_key"].n_unique(),
          "dup (match,round)", re_.height - re_.unique(["match_key", "round"]).height)
    print("player_count values:", re_.group_by("player_count").len().sort("player_count").rows())
    print("winner_team_code values:", re_.group_by("winner_team_code").len().sort("winner_team_code").rows())
    rs = rd(label, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
    fe = rs.filter(pl.col("event_type") == "round_freeze_end")
    print("freeze_end rows", fe.height, "dup (match,round)", fe.height - fe.unique(["match_key", "round"]).height,
          "matches", fe["match_key"].n_unique())
    d = fe.with_columns(
        (pl.col("t_score") != pl.col("t_score_raw")).alias("t_diff"),
        (pl.col("ct_score") != pl.col("ct_score_raw")).alias("ct_diff"))
    print("t_score != t_score_raw:", d.filter("t_diff").height, " ct_score != ct_score_raw:", d.filter("ct_diff").height)
    print("sample freeze_end rows (first match):")
    mk = d["match_key"][0]
    print(d.filter(pl.col("match_key") == mk).sort("round").select(["round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).head(30))
    pi = rd(label, "player_info", ["match_key", "round", "player_id", "team_code"])
    cnt = pi.group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("np"),
                                                  (pl.col("team_code") == 2).sum().alias("n2"),
                                                  (pl.col("team_code") == 3).sum().alias("n3"))
    print("player_info (match,round) groups", cnt.height, " np dist:", cnt.group_by("np").len().sort("np").rows()[:8])
    print("valid 5v5 groups (np=10,n2=5,n3=5):", cnt.filter((pl.col("np") == 10) & (pl.col("n2") == 5) & (pl.col("n3") == 5)).height)
    print("team_code values:", pi.group_by("team_code").len().sort("team_code").rows())
