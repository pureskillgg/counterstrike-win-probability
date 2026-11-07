r"""Verifier D4-4: value probe for round_state / bomb_state / round_end (2024 tome, projected columns only).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_01_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")


def scan(ch, cols):
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0] if ch != "header" else BASE / "header"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.scan_parquet(files).select(cols)


rs = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                          "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
print("round_state event_type counts:")
print(rs.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect())
print("round_state phase counts:")
print(rs.group_by("phase").agg(pl.len().alias("n")).sort("n", descending=True).collect())

be = scan("bomb_state", ["match_key", "round", "tick", "event_type"])
print("bomb_state event_type counts:")
print(be.group_by("event_type").agg(pl.len().alias("n")).sort("n", descending=True).collect())

re_ = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
print("round_end winner_team_code x win_reason_code (all rows):")
print(re_.group_by(["winner_team_code", "win_reason_code"]).agg(pl.len().alias("n"))
      .sort(["winner_team_code", "win_reason_code"]).collect())
print("round_end rows per (match,round) distribution:")
dup = (re_.group_by(["match_key", "round"]).agg(pl.len().alias("k"))
       .group_by("k").agg(pl.len().alias("pairs")).sort("k").collect())
print(dup)

sample = (rs.filter(pl.col("match_key").str.contains("2024/02/12/01648186"))
          .filter(pl.col("event_type").str.contains("(?i)freeze"))
          .select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"])
          .sort(["round", "tick"]).collect())
print("sample match 01648186 freeze-type rows (first 40):")
print(sample.head(40))
