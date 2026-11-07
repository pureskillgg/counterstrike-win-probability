"""Verifier D3-5, probe 2: per-match timelines for a few 2024 matches (min round = 1 and min round = 2).

Read only. Column-projected reads of small tables. Prints to stdout.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = g_read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                            "t_score_raw", "ct_score_raw"])
RST = g_read("round_start", ["match_key", "round", "tick"])
BS = g_read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])

mins = RE.group_by("match_key").agg(pl.col("round").min().alias("mn")).sort("match_key")
m_min2 = mins.filter(pl.col("mn") == 2)["match_key"].to_list()
day = pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)
print("min2 by day:", mins.filter(pl.col("mn") == 2).with_columns(day.alias("d")).group_by("d").len().sort("d").rows())

# choose: 2 day-01 min2 matches, 2 day-12 min2 matches, 2 day-12 min1 matches
d01 = [m for m in m_min2 if "/csds/2024/02/01/" in m or "/02/01/" in m][:2]
d12_min2 = [m for m in m_min2 if "/02/12/" in m][:2]
d12_min1 = mins.filter((pl.col("mn") == 1) & pl.col("match_key").str.contains("/02/12/"))["match_key"].to_list()[:2]
print("\nsample day01 min2:", d01)
print("sample day12 min2:", d12_min2)
print("sample day12 min1:", d12_min1)

for m in d01 + d12_min2 + d12_min1:
    print("\n" + "=" * 120)
    print("MATCH", m)
    re_ = RE.filter(pl.col("match_key") == m).sort("tick").select(["round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    print("round_end:")
    print(re_.rows())
    fe = RS.filter((pl.col("match_key") == m) & (pl.col("event_type") == "round_freeze_end")).sort("tick")
    print("round_state freeze_end (round, tick, t, ct, t_raw, ct_raw):")
    print(fe.select(["round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).rows())
    rst = RST.filter(pl.col("match_key") == m).sort("tick")["round"].to_list()
    print("round_start rounds:", rst)
    rsend = RS.filter((pl.col("match_key") == m) & (pl.col("event_type") == "round_end")).sort("tick")
    print("round_state round_end rows (round, tick, t, ct):", rsend.select(["round", "tick", "t_score", "ct_score"]).rows())
    bs = BS.filter((pl.col("match_key") == m) & pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).sort("tick")
    print("bomb events (round, tick, event, team):", bs.select(["round", "tick", "event_type", "player_team_code"]).rows())
