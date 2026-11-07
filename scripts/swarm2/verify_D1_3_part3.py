"""verify_D1_3 part 3: exact per-build x round-bin counts for three independent references, and the
per-match breakdown for build 9961 round 13. Read-only on the 2024 tome.

References (none of them is round_end.winner_team_code):
  R1 bomb side: bomb_defused -> 3, bomb_exploded -> 2 (bomb rounds only, player_count == 20)
  R2 elimination side: all 5 CT players dead before round_end -> 2 (T won); all 5 T players dead
     and no bomb_planted in the round -> 3 (CT won). Roster = player_info of the same round,
     death tick <= round_end tick. Requires 5 v 5 roster (10 players).
  R3 cross-round score side: t_score or ct_score rose by exactly 1 between the first round_state
     row of round r and the first row of round r+1, and the other side did not change.

ASSUMPTIONS:
  C1 build era = header.build_num (9842 vs 9961).
  C2 round bins: 9, 13, 25, 28 (claim rounds), 14-24, other.
  C3 player_count == 20 only.
  C4 a 'disagreement' is a reference side != round_end.winner_team_code.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)
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


def rbin_expr():
    return (pl.when(pl.col("round").is_in(CLAIM_ROUNDS)).then(pl.col("round").cast(pl.Utf8))
            .when(pl.col("round").is_between(14, 24)).then(pl.lit("14-24"))
            .otherwise(pl.lit("other")).alias("rbin"))


RE = (rd("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
      .unique(["match_key", "round"], keep="first", maintain_order=True))
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"])
RST = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
PI = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
PD = rd("player_death", ["match_key", "round", "tick", "player_id_fixed"])
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
HD = (pl.concat([pl.read_parquet(f, columns=["key", "build_num", "match_date"]) for f in hf], how="vertical_relaxed")
      .rename({"key": "match_key"}).with_columns(pl.col("build_num").cast(pl.Utf8)))

# R1 bomb side
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
R1 = outc.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side")
).select(["match_key", "round", "bomb_side"])
planted_any = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).unique()
               .with_columns(pl.lit(True).alias("planted")))

# R2 elimination side
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

# R3 cross-round score side
fb = RST.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"))
nx = fb.with_columns((pl.col("round") - 1).alias("round")).rename({"t0": "tn", "c0": "cn"})
R3 = (fb.join(nx, on=["match_key", "round"], how="inner")
      .with_columns(((pl.col("tn") - pl.col("t0")) == 1).alias("a"), ((pl.col("cn") - pl.col("c0")) == 1).alias("b"))
      .with_columns(pl.when(pl.col("a") & ~pl.col("b")).then(pl.lit(2))
                      .when(pl.col("b") & ~pl.col("a")).then(pl.lit(3))
                      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))
      .select(["match_key", "round", "score_side"]))

BASEF = (RE.filter(pl.col("player_count") == 20)
         .join(HD.select(["match_key", "build_num"]), on="match_key", how="left")
         .with_columns(rbin_expr())
         .join(R1, on=["match_key", "round"], how="left")
         .join(R2, on=["match_key", "round"], how="left")
         .join(R3, on=["match_key", "round"], how="left"))
print("p20 rounds in round_end:", BASEF.height, " build missing:", BASEF.filter(pl.col("build_num").is_null()).height)

# ------------------------------------------------------------------ 1. per build x bin counts
hdr("1. PER BUILD x ROUND BIN: rounds, bomb-observable mismatch, elimination-derived disagreement, score-side disagreement")
T1 = BASEF.group_by(["build_num", "rbin"]).agg(
    pl.len().alias("rounds"),
    pl.col("bomb_side").is_not_null().sum().alias("bomb_n"),
    ((pl.col("bomb_side") != pl.col("winner_team_code")) & pl.col("bomb_side").is_not_null()).sum().alias("bomb_mm"),
    pl.col("elim_side").is_not_null().sum().alias("elim_n"),
    ((pl.col("elim_side") != pl.col("winner_team_code")) & pl.col("elim_side").is_not_null()).sum().alias("elim_ne_winner"),
    pl.col("score_side").is_not_null().sum().alias("score_n"),
    ((pl.col("score_side") != pl.col("winner_team_code")) & pl.col("score_side").is_not_null()).sum().alias("score_ne_winner"),
).sort(["build_num", "rbin"])
print(T1)
for c_ in ["bomb", "elim", "score"]:
    T1 = T1.with_columns((pl.col(f"{c_}_" + ("mm" if c_ == "bomb" else "ne_winner") ) / pl.col(f"{c_}_n")).round(4).alias(f"{c_}_rate"))
print(T1.select(["build_num", "rbin", "bomb_n", "bomb_mm", "bomb_rate", "elim_n", "elim_ne_winner", "elim_rate",
                 "score_n", "score_ne_winner", "score_rate"]))

# ------------------------------------------------------------------ 2. elim crosstab for round 13 and 25/28
hdr("2. ELIMINATION SIDE VS ROUND_END WINNER, ROUND 13 / 25 / 28 / 9 BY BUILD (derivable rounds only)")
print(BASEF.filter(pl.col("elim_side").is_not_null() & pl.col("round").is_in([9, 13, 25, 28]))
      .group_by(["build_num", "round", "elim_side", "winner_team_code"]).len()
      .sort(["build_num", "round", "elim_side", "winner_team_code"]))

# ------------------------------------------------------------------ 3. per match, build 9961, round 13
hdr("3. BUILD 9961: PER-MATCH STATUS OF ROUND 13 (elimination-derivable round)")
r13 = BASEF.filter((pl.col("round") == 13) & (pl.col("build_num") == "9961")).select(
    ["match_key", "winner_team_code", "elim_side", "bomb_side", "score_side"])
n_matches_9961 = HD.filter(pl.col("build_num") == "9961").height
print("build 9961 matches:", n_matches_9961, "  round-13 rows p20:", r13.height)
print("round-13 rows with elim-derivable side:", r13.filter(pl.col("elim_side").is_not_null()).height)
print("round-13 rows with bomb side:", r13.filter(pl.col("bomb_side").is_not_null()).height)
print("round-13 rows with cross-round score side:", r13.filter(pl.col("score_side").is_not_null()).height)
print("round-13 elim side == winner:", r13.filter(pl.col("elim_side").is_not_null() & (pl.col("elim_side") == pl.col("winner_team_code"))).height,
      " elim side != winner:", r13.filter(pl.col("elim_side").is_not_null() & (pl.col("elim_side") != pl.col("winner_team_code"))).height)
print("round-13 score side == winner:", r13.filter(pl.col("score_side").is_not_null() & (pl.col("score_side") == pl.col("winner_team_code"))).height,
      " score side != winner:", r13.filter(pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("winner_team_code"))).height)
print("round-13 bomb side == winner:", r13.filter(pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") == pl.col("winner_team_code"))).height,
      " bomb side != winner:", r13.filter(pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).height)
# three-way agreement where all three exist
tw = r13.filter(pl.col("elim_side").is_not_null() & pl.col("score_side").is_not_null())
print("rows with elim AND score side:", tw.height, " elim==score:", tw.filter(pl.col("elim_side") == pl.col("score_side")).height)
print("rows with elim AND score side, winner==elim:", tw.filter(pl.col("winner_team_code") == pl.col("elim_side")).height)
print("rows with elim AND score side, winner==score:", tw.filter(pl.col("winner_team_code") == pl.col("score_side")).height)

# matches in build 9961: does every round-13 row with a reference disagree with winner?
m13 = (BASEF.filter((pl.col("round") == 13) & (pl.col("build_num") == "9961"))
       .with_columns(((pl.col("elim_side") != pl.col("winner_team_code")) & pl.col("elim_side").is_not_null()).alias("ne_elim"))
       .group_by("match_key").agg(pl.col("ne_elim").any().alias("any_ne_elim"),
                                  pl.col("elim_side").is_not_null().any().alias("has_elim")))
print("9961 matches with round 13 elim-derivable:", m13.filter(pl.col("has_elim")).height,
      " of which any elim disagreement at 13:", m13.filter(pl.col("has_elim") & pl.col("any_ne_elim")).height)
print("9961 matches with no round 13 elim-derivable row:", m13.filter(~pl.col("has_elim")).height)

# ------------------------------------------------------------------ 4. residual disagreements in rounds 14-24 and other (build 9961)
hdr("4. RESIDUAL ELIMINATION DISAGREEMENTS OUTSIDE 9/13/25/28, BUILD 9961: which matches?")
resid = (BASEF.filter((pl.col("build_num") == "9961") & pl.col("elim_side").is_not_null()
                      & (pl.col("elim_side") != pl.col("winner_team_code")) & ~pl.col("round").is_in(CLAIM_ROUNDS))
         .select(["match_key", "round", "elim_side", "winner_team_code", "bomb_side"]))
print("residual rows:", resid.height, " distinct matches:", resid["match_key"].n_unique())
print(resid.group_by("round").len().sort("round"))
# do the residual matches also have a round-13 inversion?
rm = resid.select("match_key").unique()
inv13 = (BASEF.filter((pl.col("round") == 13) & (pl.col("build_num") == "9961") & pl.col("elim_side").is_not_null()
                      & (pl.col("elim_side") != pl.col("winner_team_code"))).select("match_key").unique())
print("residual matches that also have a round-13 elim inversion:", rm.join(inv13, on="match_key", how="inner").height,
      " of", rm.height)
print(resid.head(12))

# ------------------------------------------------------------------ 5. build 9842 control
hdr("5. BUILD 9842 CONTROL: any elimination or score disagreement at all?")
c9842 = BASEF.filter(pl.col("build_num") == "9842")
print("elim-derivable rows:", c9842.filter(pl.col("elim_side").is_not_null()).height,
      " disagreements:", c9842.filter(pl.col("elim_side").is_not_null() & (pl.col("elim_side") != pl.col("winner_team_code"))).height)
print("score-side rows:", c9842.filter(pl.col("score_side").is_not_null()).height,
      " disagreements:", c9842.filter(pl.col("score_side").is_not_null() & (pl.col("score_side") != pl.col("winner_team_code"))).height)
print("bomb rows:", c9842.filter(pl.col("bomb_side").is_not_null()).height,
      " disagreements:", c9842.filter(pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).height)

print("\nDONE part3")
