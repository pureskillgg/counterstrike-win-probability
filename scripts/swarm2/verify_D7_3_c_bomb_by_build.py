"""Verify D7-3 (independent): bomb-based round-13 mismatch (the defect's own test) split by header build_num,
by raw key-path day (batch), and by UTC match date. Recomputed from basic_round_end and basic_bomb_state only.
Definitions (same as the defect, scripts/swarm/e2_side_check_r13.py): rounds with player_count == 20 and a
bomb_defused or bomb_exploded row. bomb_side = 3 if defused else 2. mismatch = bomb_side != winner_team_code.
ASSUMPTION: a round with both defused and exploded rows is reported separately (none expected).
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))

H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "match_date", "build_num", "map_name"]).rename({"key": "match_key"})
H = H.with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
                   pl.col("match_key").str.extract(r"csds/(\d{4}/\d\d/\d\d)/", 1).alias("key_day"))
H = H.with_columns(pl.col("mdt").dt.strftime("%m-%d").alias("utc_day"))

RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"]) for f in files(BASE / f"basic_round_end{SUF}")]).collect()
BS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"]) for f in files(BASE / f"basic_bomb_state{SUF}")]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
ev = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
print("rounds with outcome rows:", ev.height, "| with both defused and exploded:", ev.filter(pl.col("evs").list.len() > 1).height)
ev = ev.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))

J = (RE.filter(pl.col("player_count") == 20).join(ev, on=["match_key", "round"], how="inner")
       .join(H.select(["match_key", "build_num", "key_day", "utc_day", "map_name", "mdt"]), on="match_key", how="left")
       .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                     pl.when(pl.col("round") == 13).then(pl.lit("r13")).otherwise(pl.lit("other")).alias("rbin")))
print("bomb rounds (20 players):", J.height, "| mismatches:", int(J["mm"].sum()), "| null build:", J.filter(pl.col("build_num").is_null()).height)
print("\nall rounds x build:", J.group_by(["rbin", "build_num", "mm"]).len().sort(["rbin", "build_num", "mm"]).rows())
print("\nround 13 x build (n, mm, rate):")
r13 = J.filter(pl.col("round") == 13)
print(r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), pl.col("mm").mean().round(4).alias("rate")).sort("build_num"))
print("\nround 13 x key_day x build (n, mm):")
print(r13.group_by(["key_day", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["key_day", "build_num"]).rows())
print("\nnon-13 rounds mismatches by build x round (only mm rows):")
oth = J.filter((pl.col("round") != 13) & pl.col("mm"))
print(oth.group_by(["build_num", "round"]).len().sort(["build_num", "round"]).rows())
print("non-13 rounds: n by build:", J.filter(pl.col("round") != 13).group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).rows())

# Decisive alternative test: batch (raw key-day) vs build inside the SAME key-days 12..25
mix = r13.filter(pl.col("key_day") >= "2024/02/12")
print("\nWITHIN-BATCH: round 13 rows, key_day >= 2024/02/12, by build (n, mm, rate):")
print(mix.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"), pl.col("mm").mean().round(4).alias("rate")).sort("build_num"))
pre = r13.filter(pl.col("key_day") < "2024/02/12")
print("round 13, key_day < 2024/02/12, by build:", pre.group_by("build_num").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).rows())

print("\nround 13 by UTC match date (mm, n) for build 9842 after 2024-02-06:")
print(r13.filter((pl.col("build_num") == 9842) & (pl.col("mdt") >= pl.datetime(2024, 2, 6))).select(["mdt", "key_day", "winner_team_code", "bomb_side", "mm", "map_name"]).sort("mdt").with_columns(pl.col("mdt").cast(pl.Utf8)).rows())
print("round 13 bomb rows, build 9961, mdt before 2024-02-12:", r13.filter((pl.col("build_num") == 9961) & (pl.col("mdt") < pl.datetime(2024, 2, 12))).height)

print("\nround 13 mismatch by map x build (n, mm):")
print(r13.group_by(["map_name", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["map_name", "build_num"]).rows())
print("\nround 13 mismatch: 9961 rows with NO mismatch (match ids):")
print(r13.filter((pl.col("build_num") == 9961) & (~pl.col("mm"))).select(["match_key", "key_day", "mdt", "map_name", "winner_team_code", "bomb_side", "evs"]).with_columns(pl.col("match_key").str.slice(0, 40), pl.col("mdt").cast(pl.Utf8)).rows())
print("\nround 13 mismatch: 9842 rows WITH mismatch (count):", r13.filter((pl.col("build_num") == 9842) & pl.col("mm")).height)

# Mismatch direction: winner vs bomb side
print("\nmismatch direction (round 13, build 9961): winner_team_code x bomb_side:")
print(r13.filter((pl.col("build_num") == 9961) & pl.col("mm")).group_by(["winner_team_code", "bomb_side"]).len().rows())
print("round 13 9961 all rows: winner x bomb:", r13.filter(pl.col("build_num") == 9961).group_by(["winner_team_code", "bomb_side"]).len().rows())
