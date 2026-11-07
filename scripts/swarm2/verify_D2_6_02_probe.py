"""verify D2-6: probe final-score row, death side field, and round-1 team identity vs header starters."""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
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
    rs = scan(label, "round_state").filter(pl.col("event_type") == "cs_win_panel_match").select(
        ["match_key", "round", "tick", "t_score", "ct_score"]).collect()
    print("cs_win_panel_match rows:", rs.height, "unique mk:", rs["match_key"].n_unique())
    print(rs.head(5))
    h = scan(label, "header").select(["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
    j = rs.join(h, left_on="match_key", right_on="key", how="left")
    print("panel vs header (t_score==t_starters_final, ct_score==ct_starters_final):",
          j.filter((pl.col("t_score") == pl.col("t_starters_score_final")) & (pl.col("ct_score") == pl.col("ct_starters_score_final"))).height,
          "of", j.height)
    print("panel vs header swapped (t_score==ct_starters_final):",
          j.filter((pl.col("t_score") == pl.col("ct_starters_score_final")) & (pl.col("ct_score") == pl.col("t_starters_score_final"))).height)
    pdth = scan(label, "player_death").select(["match_key", "round", "player_id", "player_team_code"]).collect()
    print("player_death rows", pdth.height, "team_code counts", pdth.group_by("player_team_code").len().sort("player_team_code").rows())
    pi = scan(label, "player_info").select(["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"]).collect()
    jj = pdth.join(pi, on=["match_key", "round", "player_id"], how="left")
    print("death rows with membership:", jj.filter(pl.col("team_code").is_not_null()).height,
          " player_team_code == membership team_code:", jj.filter(pl.col("team_code") == pl.col("player_team_code")).height,
          " differ:", jj.filter((pl.col("team_code").is_not_null()) & (pl.col("team_code") != pl.col("player_team_code"))).height)
    print(jj.filter((pl.col("team_code").is_not_null()) & (pl.col("team_code") != pl.col("player_team_code"))).head(5))
