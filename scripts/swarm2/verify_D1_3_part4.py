"""verify_D1_3 part 4: close residual questions. Read-only on the 2024 tome.
  Q1 residual elimination disagreements outside 9/13/25/28 (build 9961): does the cross-round score side agree with the winner or with the elimination side?
  Q2 build 9961 round 13: round_end reason distribution for rows where score side == winner versus != winner.
  Q3 build x match_date range.
  Q4 side-swap structure: build 9961 elimination inversion rate by round around the swap boundaries.
Reuses the reference definitions of verify_D1_3_part3.py (R2 elimination, R3 cross-round score).
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
CLAIM_ROUNDS = [9, 13, 25, 28]


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True))
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
PD = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"])
RST = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"])
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
HD = (pl.concat([pl.read_parquet(f, columns=["key", "build_num", "match_date"]) for f in hf], how="vertical_relaxed")
      .rename({"key": "match_key"}).with_columns(pl.col("build_num").cast(pl.Utf8)))

planted_any = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique()
               .with_columns(pl.lit(True).alias("planted")))
roster = (PI.with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
          .filter(pl.col("pid").is_not_null()).select(["match_key", "round", "pid", "team_code"])
          .unique(["match_key", "round", "pid"]))
deaths = (PD.with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
          .filter(pl.col("pid").is_not_null()).select(["match_key", "round", "pid", "tick"])
          .sort("tick").unique(["match_key", "round", "pid"], keep="first"))
rr = (roster.join(deaths, on=["match_key", "round", "pid"], how="left")
      .join(RE.select(["match_key", "round", pl.col("tick").alias("end_tick")]), on=["match_key", "round"], how="inner")
      .with_columns(((pl.col("tick").is_not_null()) & (pl.col("tick") <= pl.col("end_tick"))).alias("dead")))
alv = rr.group_by(["match_key", "round"]).agg(
    (pl.col("team_code") == 2).sum().alias("nT"), (pl.col("team_code") == 3).sum().alias("nCT"),
    ((pl.col("team_code") == 2) & ~pl.col("dead")).sum().alias("T_alive"),
    ((pl.col("team_code") == 3) & ~pl.col("dead")).sum().alias("CT_alive"))
alv = alv.join(planted_any, on=["match_key", "round"], how="left").with_columns(pl.col("planted").fill_null(False))
R2 = alv.filter((pl.col("nT") == 5) & (pl.col("nCT") == 5)).with_columns(
    pl.when((pl.col("CT_alive") == 0) & (pl.col("T_alive") > 0)).then(pl.lit(2))
      .when((pl.col("T_alive") == 0) & (pl.col("CT_alive") > 0) & (~pl.col("planted"))).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_side")).select(["match_key", "round", "elim_side"])
fb = RST.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"))
nx = fb.with_columns((pl.col("round") - 1).alias("round")).rename({"t0": "tn", "c0": "cn"})
R3 = (fb.join(nx, on=["match_key", "round"], how="inner")
      .with_columns(((pl.col("tn") - pl.col("t0")) == 1).alias("a"), ((pl.col("cn") - pl.col("c0")) == 1).alias("b"))
      .with_columns(pl.when(pl.col("a") & ~pl.col("b")).then(pl.lit(2))
                      .when(pl.col("b") & ~pl.col("a")).then(pl.lit(3))
                      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))
      .select(["match_key", "round", "score_side"]))
X = (RE.filter(pl.col("player_count") == 20)
     .join(HD.select(["match_key", "build_num", "match_date"]), on="match_key", how="left")
     .join(R2, on=["match_key", "round"], how="left")
     .join(R3, on=["match_key", "round"], how="left"))

hdr("Q1. RESIDUAL ELIMINATION DISAGREEMENTS OUTSIDE 9/13/25/28 (build 9961): score side versus winner and versus elimination")
res = X.filter((pl.col("build_num") == "9961") & pl.col("elim_side").is_not_null()
               & (pl.col("elim_side") != pl.col("winner_team_code")) & ~pl.col("round").is_in(CLAIM_ROUNDS))
print("residual rows:", res.height)
print(res.with_columns(
    pl.when(pl.col("score_side").is_null()).then(pl.lit("score_null"))
      .when(pl.col("score_side") == pl.col("winner_team_code")).then(pl.lit("score==winner"))
      .when(pl.col("score_side") == pl.col("elim_side")).then(pl.lit("score==elim"))
      .otherwise(pl.lit("other")).alias("rel")).group_by("rel").len().sort("len", descending=True))
print("residual rows by round:", res.group_by("round").len().sort("round").to_dicts())
print("residual rows by reason:", res.group_by("win_reason_message").len().sort("len", descending=True).to_dicts())

hdr("Q2. BUILD 9961 ROUND 13: ROUND_END REASON, SCORE SIDE == WINNER versus != WINNER")
r13 = X.filter((pl.col("build_num") == "9961") & (pl.col("round") == 13) & pl.col("score_side").is_not_null())
r13 = r13.with_columns(pl.when(pl.col("score_side") == pl.col("winner_team_code")).then(pl.lit("agree"))
                         .otherwise(pl.lit("disagree")).alias("rel"))
print(r13.group_by(["rel", "win_reason_message"]).len().sort(["rel", "len"], descending=[False, True]))
print("elim-derivable in agree group:", r13.filter((pl.col("rel") == "agree") & pl.col("elim_side").is_not_null()).height)

hdr("Q3. BUILD x MATCH_DATE RANGE")
HDd = HD.with_columns(pl.col("match_date").str.slice(0, 10).alias("d"))
print(HDd.group_by("build_num").agg(pl.col("d").min().alias("min_date"), pl.col("d").max().alias("max_date"),
                                    pl.len().alias("matches")).sort("build_num").to_dicts())
print("matches dated >= 2024-02-07 by build:",
      HDd.filter(pl.col("d") >= "2024-02-07").group_by("build_num").len().sort("build_num").to_dicts())
print("matches dated < 2024-02-07 by build:",
      HDd.filter(pl.col("d") < "2024-02-07").group_by("build_num").len().sort("build_num").to_dicts())

hdr("Q4. ELIMINATION INVERSION RATE BY ROUND, BUILD 9961, AROUND SIDE-SWAP BOUNDARIES")
ex = X.filter((pl.col("build_num") == "9961") & pl.col("elim_side").is_not_null())
ex = ex.with_columns((pl.col("elim_side") != pl.col("winner_team_code")).alias("inv"))
print(ex.filter(pl.col("round").is_in([8, 9, 10, 12, 13, 14, 24, 25, 26, 27, 28, 29])).group_by("round")
      .agg(pl.len().alias("n"), pl.col("inv").sum().alias("inv"), pl.col("inv").mean().round(3).alias("rate"))
      .sort("round"))

print("\nDONE part4")
