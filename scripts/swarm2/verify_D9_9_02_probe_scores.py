"""verify D9-9, step 2: probe round_state score fields and bomb event types (read-only, 2024-02 tome).

Looks at: round_state event types, the score fields around round boundaries for one build-9961
match and one build-9842 match, and bomb_state event types. Also tests the header final scores
against the recorded round_end winners. Projection-only lazy scans; small output.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


H = scan("header", ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"])
H = H.with_columns(pl.col("key").alias("match_key"))
print("header rows", H.height, "| build counts:", H.group_by("build_num").len().sort("build_num").rows())

RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                          "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
print("round_state rows", RS.height)
print("round_state event types:", RS.group_by("event_type").len().sort("len", descending=True).rows()[:30])

BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
print("bomb_state event types:", BS.group_by("event_type").len().sort("len", descending=True).rows())

RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message"])
print("round_end rows", RE.height)

# one example match per build: rounds 12..14
for b in [9842, 9961]:
    keys = H.filter(pl.col("build_num") == b).select("match_key").head(1)["match_key"].to_list()
    if not keys:
        print("no match for build", b)
        continue
    k = keys[0]
    print(f"\n== example build {b}: {k}")
    print("header finals:", H.filter(pl.col("match_key") == k).select(
        ["t_starters_score_final", "ct_starters_score_final"]).rows())
    re_k = RE.filter(pl.col("match_key") == k).sort("round")
    print("round_end (round, tick, winner, msg):")
    print(re_k.select(["round", "tick", "winner_team_code", "win_reason_message"]).head(16))
    rs_k = RS.filter((pl.col("match_key") == k) & pl.col("event_type").is_in(
        ["round_end", "round_freeze_end", "round_start"])).sort(["tick"])
    print("round_state rows around rounds (first 40):")
    print(rs_k.select(["round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).head(40))
