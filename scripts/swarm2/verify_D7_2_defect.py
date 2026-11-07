"""verify D7-2 (part 2): does build_num, match_date (time zone) or path day explain the OPEN DEFECT itself?
Defect definition (same as scripts/swarm/e2_side_check_r13.py): a round with a bomb outcome (bomb_defused -> side 3,
bomb_exploded -> side 2) whose winner_team_code differs from the bomb-derived side is a mismatch.
Population: all 2024 tome rounds with player_count == 20 and a bomb outcome. Raw inputs only:
  basic_round_end, basic_bomb_state, header (match_date, build_num).
Memory: bomb_state is filtered lazily to two event types and three columns before collect.
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
H = H.with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"),
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("path_day"))
H = H.with_columns(pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7, 0, 24, 22)).then(pl.lit("A: <00:24:22"))
                   .when(pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18)).then(pl.lit("B: gap"))
                   .otherwise(pl.lit("C: >=01:01:18")).alias("zone"))

RE = (pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count"])
                 for f in files(BASE / f"basic_round_end{SUF}")], how="vertical_relaxed")
      .filter(pl.col("player_count") == 20))
RE = RE.unique(subset=["match_key", "round"], keep="first")

BS = (pl.concat([pl.scan_parquet(f).filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
                   .select(["match_key", "round", "event_type"]) for f in files(BASE / f"basic_bomb_state{SUF}")])
      .collect())
outc = (BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
both = outc.filter(pl.col("evs").list.len() > 1).height
print("bomb outcome keys:", outc.height, "| keys with both defused and exploded:", both)

J = (RE.join(outc, on=["match_key", "round"], how="inner")
     .with_columns(
         pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                   pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
                     .otherwise(pl.lit("other")).alias("rbin"))
     .join(H, on="match_key", how="left"))
print("rounds with bomb outcome (20p):", J.height, "| mismatches:", J.filter(pl.col("mm")).height,
      "| build_num null:", J.filter(pl.col("build_num").is_null()).height)

print("\nmismatch by round bin x build_num (n, mismatches):")
print(J.group_by(["rbin", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["rbin", "build_num"]).rows())

print("\nmismatch by round bin x time zone (mdt cut 01:01:18 Feb 7):")
print(J.group_by(["rbin", "zone"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["rbin", "zone"]).rows())

r13 = J.filter(pl.col("rbin") == "r13")
print("\nround 13: build x time zone x mismatch")
print(r13.group_by(["build_num", "zone", "mm"]).len().sort(["build_num", "zone", "mm"]).rows())

print("\nround 13: path day x build_num -> (n, mismatches)")
print(r13.group_by(["path_day", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
      .sort(["path_day", "build_num"]).rows())

# path-day mismatch rate vs build share (is path day only a mixture of builds?)
pdr = (r13.group_by("path_day").agg(pl.len().alias("n"), pl.col("mm").mean().alias("mm_rate"),
                                    (pl.col("build_num") == 9961).mean().alias("share_9961_in_r13"))
       .sort("path_day"))
print("\nround 13 by path day: n, mismatch rate, share of build 9961 among these r13 rows")
print(pdr.rows())

print("\nmismatch rows NOT in build 9961 (any round):", J.filter(pl.col("mm") & (pl.col("build_num") != 9961)).height)
print("build 9961 rows that are NOT mismatches (any round):", J.filter((~pl.col("mm")) & (pl.col("build_num") == 9961)).height)
print("list of non-9961 mismatches:")
print(J.filter(pl.col("mm") & (pl.col("build_num") != 9961)).select(["match_key", "round", "mdt", "build_num", "winner_team_code", "bomb_side"])
      .with_columns(pl.col("match_key").str.slice(0, 45)).rows())
print("\n9961 non-mismatch rows (round bin counts):")
print(J.filter((~pl.col("mm")) & (pl.col("build_num") == 9961)).group_by("rbin").len().rows())

# the single match whose match_date is a nanosecond stamp and build 9842 (claim's one mdt error)
print("\nrows for match m_98cefeb6 (the claim's single mdt error):")
print(J.filter(pl.col("match_key").str.starts_with("csds/2024/02/12/m_partial")).select(
    ["round", "mdt", "build_num", "winner_team_code", "bomb_side", "mm", "rbin", "zone"]).sort("round").rows())


# ================= map confound check on the defect population =================
HM2 = (pl.concat([pl.read_parquet(f, columns=["key", "map_name"]) for f in files(BASE / "header")], how="vertical_relaxed")
       .rename({"key": "match_key"}))
Jm = J.join(HM2, on="match_key", how="left")
print("\ndefect population, round 13, by map x build (n, mismatches):")
print(Jm.filter(pl.col("rbin") == "r13").group_by(["map_name", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm"))
      .sort(["map_name", "build_num"]).rows())
