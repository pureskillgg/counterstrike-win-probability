"""verify_D1_3 part 5: (i) independent check of the D1 'no halftime flip' hypothesis; (ii) characterise the
161 build-9961 round-13 rows where the cross-round score side agrees with round_end.winner_team_code.
Read-only on the 2024 tome.

  P1 per match team-code flip at 12 -> 13 from player_info (same player_id_fixed in rounds 12 and 13).
  P2 planter check in mismatch matches: planter team in bomb_state and player_info at round 13 vs round 12.
  P3 the 161 'agree' rows: map, match date, and the state of rounds 25 and 28 in the same match.

ASSUMPTIONS:
  E1 flip fraction = changed team_code / covered players (present in both rounds).
  E2 flip_full >= 0.9, no_flip <= 0.1, otherwise partial.
  E3 'agree' row = build 9961, round 13, player_count 20, cross-round score side non-null and == winner_team_code.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True))
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"]).with_columns(
    pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_id_fixed", "player_team_code"])
RST = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
HD = (pl.concat([pl.read_parquet(f, columns=["key", "build_num", "map_name", "match_date"]) for f in hf], how="vertical_relaxed")
      .rename({"key": "match_key"}).with_columns(pl.col("build_num").cast(pl.Utf8)))

# P1 flip 12 -> 13
hdr("P1. TEAM-CODE FLIP 12 -> 13 PER MATCH (player_info), by build")
a = PI.filter(pl.col("round") == 12).select(["match_key", "pid", pl.col("team_code").alias("t12")])
b = PI.filter(pl.col("round") == 13).select(["match_key", "pid", pl.col("team_code").alias("t13")])
fl = (a.join(b, on=["match_key", "pid"], how="inner")
      .with_columns((pl.col("t12") != pl.col("t13")).alias("flip"))
      .group_by("match_key").agg(pl.len().alias("n"), pl.col("flip").mean().alias("ff"))
      .join(HD.select(["match_key", "build_num"]), on="match_key", how="left")
      .with_columns(pl.when(pl.col("n") < 6).then(pl.lit("not_evaluable"))
                      .when(pl.col("ff") >= 0.9).then(pl.lit("flip_full"))
                      .when(pl.col("ff") <= 0.1).then(pl.lit("no_flip"))
                      .otherwise(pl.lit("partial")).alias("cls")))
print(fl.group_by(["build_num", "cls"]).len().sort(["build_num", "cls"]))

# P2 planter check
hdr("P2. PLANTER CHECK: planter team in bomb_state vs player_info at rounds 12 and 13")
pl_ = (BS.filter(pl.col("event_type") == "bomb_planted")
       .group_by(["match_key", "round"]).agg(pl.col("player_id_fixed").cast(pl.Int64, strict=False).first().alias("pid"),
                                             pl.col("player_team_code").first().alias("bs_team"))
       .join(HD.select(["match_key", "build_num"]), on="match_key", how="left"))
pl13 = pl_.filter(pl.col("round") == 13).join(
    PI.filter(pl.col("round") == 13).select(["match_key", "pid", pl.col("team_code").alias("pi13")]), on=["match_key", "pid"], how="left")
pl13 = pl13.join(PI.filter(pl.col("round") == 12).select(["match_key", "pid", pl.col("team_code").alias("pi12")]),
                 on=["match_key", "pid"], how="left")
print("planted at round 13 by build (bs_team, pi12, pi13):")
print(pl13.group_by(["build_num", "bs_team", "pi12", "pi13"]).len().sort(["build_num", "len"], descending=[False, True]).head(20))
print("planter at round 13 always T (bs_team == 2) where planted:",
      round(float((pl13.filter(pl.col("bs_team").is_not_null())["bs_team"] == 2).mean()), 4),
      " n =", pl13.filter(pl.col("bs_team").is_not_null()).height)

# P3 the 161 agree rows
hdr("P3. BUILD 9961 ROUND 13 AGREE ROWS (score side == winner): map, date, and rounds 25/28 in the same match")
rs = RST.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"))
nx = rs.with_columns((pl.col("round") - 1).alias("round")).rename({"t0": "tn", "c0": "cn"})
sd = (rs.join(nx, on=["match_key", "round"], how="inner")
      .with_columns(((pl.col("tn") - pl.col("t0")) == 1).alias("a"), ((pl.col("cn") - pl.col("c0")) == 1).alias("b"))
      .with_columns(pl.when(pl.col("a") & ~pl.col("b")).then(pl.lit(2))
                      .when(pl.col("b") & ~pl.col("a")).then(pl.lit(3))
                      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))
      .select(["match_key", "round", "score_side"]))
X = (RE.filter(pl.col("player_count") == 20).join(HD, on="match_key", how="left")
     .join(sd, on=["match_key", "round"], how="left")
     .with_columns((pl.col("score_side") == pl.col("winner_team_code")).alias("agree")))
x13 = X.filter((pl.col("build_num") == "9961") & (pl.col("round") == 13) & pl.col("score_side").is_not_null())
ag = x13.filter(pl.col("agree")).select("match_key")
print("agree rows:", ag.height, " disagree rows:", x13.filter(~pl.col("agree")).height)
print("agree rows by map:", x13.filter(pl.col("agree")).group_by("map_name").len().sort("len", descending=True).to_dicts())
print("share agree by map:", x13.group_by("map_name").agg(pl.len().alias("n"), pl.col("agree").sum().alias("agree"),
                                                          pl.col("agree").mean().round(3).alias("share")).sort("map_name").to_dicts())
print("agree rows by date:", x13.filter(pl.col("agree")).with_columns(pl.col("match_date").str.slice(0, 10).alias("d"))
      .group_by("d").len().sort("d").to_dicts())
# state of rounds 25 and 28 in the agree matches vs disagree matches (score side vs winner)
r2528 = X.filter((pl.col("build_num") == "9961") & pl.col("round").is_in([25, 28]) & pl.col("score_side").is_not_null()) \
    .select(["match_key", "round", "agree"]).rename({"agree": "agree_2528"})
joined = x13.select(["match_key", "agree"]).join(r2528, on="match_key", how="left")
print("round 25/28 agreement (score side == winner) in matches where round 13 agrees vs disagrees:")
print(joined.group_by(["agree", "round"]).agg(pl.len().alias("n"), pl.col("agree_2528").sum().alias("agree_2528"))
      .sort(["agree", "round"]))

print("\nDONE part5")
