"""Verifier D3-5, probe 1: schema, keys, duplicates, and the day 01-07 round-gap question on the 2024 G: tome.

Read only. Prints to stdout (not written to the log). Column-projected reads of small tables.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


hdr = pl.concat([pl.read_parquet(f, columns=["key", "match_id", "match_date", "tick_rate", "map_name",
                                             "t_starters_score_final", "ct_starters_score_final"])
                 for f in sorted(glob.glob(str(TOME / "header" / "dataframe_*")))])
print("header rows:", hdr.height, "unique keys:", hdr["key"].n_unique())
print(hdr.head(3))
print("sample keys:", hdr["key"].head(3).to_list())
print("header match_date sample:", hdr["match_date"].head(5).to_list())

RE = g_read("round_end")
print("\nround_end rows:", RE.height, "matches:", RE["match_key"].n_unique())
print("player_count dist:", RE.group_by("player_count").len().sort("player_count").rows())
print("winner_team_code dist:", RE.group_by("winner_team_code").len().sort("winner_team_code").rows())
print("win_reason_code dist:", RE.group_by(["win_reason_code", "win_reason_message"]).len().sort("win_reason_code").rows())
dups = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end (match,round) with >1 row:", dups.height)
conf = (RE.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").n_unique().alias("nw"),
                                                 pl.len().alias("n"))
        .filter(pl.col("nw") > 1))
print("round_end (match,round) with conflicting winners:", conf.height)
if dups.height:
    print(RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"]).sort(["match_key", "round", "tick"]).head(20))

RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw", "is_warmup"])
print("\nround_state rows:", RS.height, "matches:", RS["match_key"].n_unique())
print("event types:", RS.group_by("event_type").len().sort("len", descending=True).head(20).rows())
fe = RS.filter(pl.col("event_type") == "round_freeze_end")
fe_c = fe.group_by(["match_key", "round"]).len().rename({"len": "nfe"})
print("freeze_end rows per (match,round) dist:", fe_c.group_by("nfe").agg(pl.len().alias("ngroups")).sort("nfe").rows())
diff_raw = RS.filter((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw")))
print("rows where score != score_raw:", diff_raw.height)
print(diff_raw.head(5))

# day parsing
mk = RE["match_key"].unique().to_list()[:3]
print("\nmatch_key samples:", mk)
days = RE.select(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")).group_by("day").len().sort("day")
print("round_end match days (per row):", days.rows())
mdays = RE.select(["match_key", pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")]).unique("match_key").group_by("day").len().sort("day")
print("matches per day (round_end):", mdays.rows())

# round numbering
rnums = RE.group_by("match_key").agg(pl.col("round").min().alias("mn"), pl.col("round").max().alias("mx"),
                                      pl.col("round").n_unique().alias("nu"), pl.len().alias("n"))
rnums = rnums.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print("\nround min dist by day:")
print(rnums.group_by(["day", "mn"]).len().sort(["day", "mn"]).head(60).rows())
print("rows-minus-unique by day:", rnums.with_columns((pl.col("n") - pl.col("nu")).alias("dup")).group_by("day").agg(pl.col("dup").sum()).sort("day").rows())
gaps = rnums.with_columns((pl.col("mx") - pl.col("nu")).alias("gapcount"))
print("gapcount dist by day:", gaps.group_by(["day", "gapcount"]).len().sort(["day", "gapcount"]).head(80).rows())
