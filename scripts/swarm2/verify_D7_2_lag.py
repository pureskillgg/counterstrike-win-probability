"""verify D7-2 (part 3): the -429 tick-lag regime from raw ticks, by build_num and by match_date zone.
lag = bomb outcome tick (bomb_defused / bomb_exploded) minus round_end tick, same round, same match. Player_count == 20.
Also: is the claim's 'first build-9961 match' the earliest 9961 match, and does round_end have duplicate (match, round) rows?
"""
import sys, glob
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


H = (pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num"]) for f in files(BASE / "header")],
               how="vertical_relaxed").rename({"key": "match_key"}))
H = H.with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))

RE = (pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "winner_team_code", "player_count"])
                 for f in files(BASE / f"basic_round_end{SUF}")], how="vertical_relaxed")
      .filter(pl.col("player_count") == 20))
dups = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end 20p duplicate (match,round) keys:", dups.height)
if dups.height:
    dd = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"])
    print(dd.with_columns(pl.col("match_key").str.slice(0, 45)).rows())
RE = RE.unique(subset=["match_key", "round"], keep="first").select(["match_key", "round", "tick"])

BS = (pl.concat([pl.scan_parquet(f).filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
                   .select(["match_key", "round", "tick"]) for f in files(BASE / f"basic_bomb_state{SUF}")])
      .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("out_tick")).collect())

J = RE.join(BS, on=["match_key", "round"], how="inner").with_columns((pl.col("out_tick") - pl.col("tick")).alias("lag"))
J = J.join(H.select(["match_key", "build_num", "mdt"]), on="match_key", how="left")
print("\nbomb rounds with lag:", J.height)
print("lag == -429 by build_num (matches with a bomb round):")
M = J.group_by(["match_key", "build_num"]).agg((pl.col("lag") == -429).any().alias("has_429"))
print(M.group_by(["build_num", "has_429"]).len().sort(["build_num", "has_429"]).rows())
print("lag value counts for build 9842 (top 8):", J.filter(pl.col("build_num") == 9842).group_by("lag").len().sort("len", descending=True).head(8).rows())
print("lag value counts for build 9961 (top 8):", J.filter(pl.col("build_num") == 9961).group_by("lag").len().sort("len", descending=True).head(8).rows())

# earliest build-9961 matches by match_date
e = H.filter(pl.col("build_num") == 9961).sort("mdt").head(5).select(["match_key", "match_date", "build_num"])
print("\nearliest 5 build-9961 matches:")
print(e.with_columns(pl.col("match_key").str.slice(0, 45)).rows())
print("earliest build-9961 mdt:", H.filter(pl.col("build_num") == 9961)["mdt"].min())
print("latest build-9842 mdt:", H.filter(pl.col("build_num") == 9842)["mdt"].max())
print("latest build-9842 match_date raw:", H.filter(pl.col("build_num") == 9842).sort("mdt").tail(3).select(["match_key", "match_date"]).with_columns(pl.col("match_key").str.slice(0, 45)).rows())


# ================= identity check: first -429 bomb round by match_date =================
m429 = J.filter(pl.col("lag") == -429).sort("mdt").head(3).select(["match_key", "mdt", "round", "build_num"])
print("IDENTITY first -429 bomb rounds by mdt:", m429.with_columns(pl.col("match_key").str.slice(0, 45)).rows())
