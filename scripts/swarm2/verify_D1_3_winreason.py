"""verify D1-3: third-source check with round_end win_reason fields (read-only, 2024 tome).

Question: on rounds where winner_team_code disagrees with the score side (S2), which side does
the independent win_reason_message / win_reason_code point to? Also: per-match overlap of the
round-9 / 13 / 25 / 27 / 28 disagreement sets, and the affected-match count (claim: 480).

Definitions (ASSUMPTIONS):
  A1 S2 score side as in verify_D1_3_recompute.py (round_state first/last rows, +1 on one side).
  A2 win_reason-derived side is taken only from message text containing 'Terrorist'/'CT' tokens or
     a bomb word; the mapping is inferred from the data (printed), not assumed from documentation.
  A3 day groups as in the claim (d01-07 vs d12-25).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(60)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = rd("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "win_reason_message",
                      "legacy_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS = rd("bomb_state", ["match_key", "round", "event_type"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])

hdr("A. win_reason_message and win_reason_code vocabulary, with winner_team_code")
print(RE.group_by(["win_reason_code", "win_reason_message"]).len().sort("len", descending=True))
print("\nlegacy_code distribution:")
print(RE.group_by("legacy_code").len().sort("len", descending=True).head(12))

# bomb outcome
oc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
oc = oc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))

# S2 score side
pre = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
       .agg(pl.col("t_score").first().alias("tp"), pl.col("ct_score").first().alias("cp")))
pre_fallback = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tp0"), pl.col("ct_score").first().alias("cp0"))
post = (RS.filter(pl.col("event_type") == "round_officially_ended").group_by(["match_key", "round"])
        .agg(pl.col("t_score").last().alias("tq"), pl.col("ct_score").last().alias("cq")))
post_fallback = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("tq0"), pl.col("ct_score").last().alias("cq0"))
sc = (pre_fallback.join(pre, on=["match_key", "round"], how="left").join(post_fallback, on=["match_key", "round"], how="left")
      .join(post, on=["match_key", "round"], how="left"))
sc = sc.with_columns(pl.coalesce(["tp", "tp0"]).alias("tp"), pl.coalesce(["cp", "cp0"]).alias("cp"),
                     pl.coalesce(["tq", "tq0"]).alias("tq"), pl.coalesce(["cq", "cq0"]).alias("cq"))
sc = sc.with_columns(
    pl.when(((pl.col("tq") - pl.col("tp")) == 1) & ((pl.col("cq") - pl.col("cp")) == 0)).then(2)
      .when(((pl.col("cq") - pl.col("cp")) == 1) & ((pl.col("tq") - pl.col("tp")) == 0)).then(3)
      .otherwise(None).alias("S2"))

R = (RE.join(oc, on=["match_key", "round"], how="left").join(sc.select(["match_key", "round", "S2"]), on=["match_key", "round"], how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
     .with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg")))
R = R.with_columns(
    (pl.col("S2") != pl.col("winner_team_code")).alias("S2_dis"),
    (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_mm"))

hdr("B. win_reason_message crosstab with winner and S2 on the rounds that matter (d12-25, rounds 9/13/25/27/28)")
Q = R.filter(pl.col("dg") == "d12-25").filter(pl.col("round").is_in([9, 13, 25, 27, 28]))
print(Q.group_by(["round", "win_reason_message", "winner_team_code", "S2"]).len()
      .sort(["round", "len"], descending=[False, True]).head(80))

hdr("C. round-13 and round-9 disagreement: which side does win_reason agree with?")
print("(rows: d12-25 round 13 where S2 defined. Columns: win_reason_message)")
r13 = R.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25") & pl.col("S2").is_not_null())
print(r13.group_by(["S2_dis", "win_reason_message", "winner_team_code", "S2"]).len().sort(["S2_dis", "len"], descending=[False, True]).head(60))
