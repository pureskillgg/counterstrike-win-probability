"""verify D9-9, step 10: three-source comparison per round (2024-02 tome, read-only).

Sources, each computed here from the raw tables (not from the earlier scratch outputs, except the
round-level scoreboard winner which comes from verify_D9_9_03 and is re-derived in this file):
  RE  round_end.winner_team_code as recorded
  SB  scoreboard winner (round_state t_score/ct_score increments; post = cs_pre_restart)
  BO  bomb outcome: bomb_exploded -> 2 (T), bomb_defused -> 3 (CT); planted-only rounds have no BO
  DE  death/roster check: for non-bomb rounds, winner w passes if the opponents of w are all dead
      (victim side = player_death.player_team_code, roster from player_info.team_code, deduped per player)

Outputs per build x round class: agreement of RE and SB with BO (bomb rounds), pass rate of RE, SB and
the label-flipped winner under DE (non-bomb rounds), and planter side distribution.

ASSUMPTIONS (ASSUMPTION):
  - bomb outcome events are the physical outcome (not verified externally).
  - 2 = T, 3 = CT in round_end, player_info, player_death and bomb_state (same convention).
  - non-bomb DE check is only a necessary condition for an elimination win; time-outs are not elimination.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(24)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9_9verify")


def scan(name, want):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*"))))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


J = pl.read_parquet(SCR / "round_score_join.parquet").select(
    ["match_key", "round", "winner_team_code", "win_reason_message", "player_count", "build_num",
     "score_winner", "t_pre", "ct_pre", "t_post", "ct_post"]
).rename({"winner_team_code": "RE", "score_winner": "SB"})
print("rounds in join:", J.height)

BS = scan("bomb_state", ["match_key", "round", "event_type", "player_team_code"])
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
             (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
             pl.len().alias("n_outc")))
plant = (BS.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"])
         .agg(pl.col("player_team_code").first().alias("planter_side"),
              pl.len().alias("n_plant")))

PI = scan("player_info", ["match_key", "round", "player_id", "team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="last")
roster = (PI.group_by(["match_key", "round"]).agg(
    (pl.col("team_code") == 2).sum().alias("n_T"),
    (pl.col("team_code") == 3).sum().alias("n_CT"),
    pl.len().alias("n_roster")))

DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"]).unique(
    subset=["match_key", "round", "player_id"], keep="first")
dead = (DE.group_by(["match_key", "round"]).agg(
    (pl.col("player_team_code") == 2).sum().alias("T_dead"),
    (pl.col("player_team_code") == 3).sum().alias("CT_dead")))

T = (J.join(outc, on=["match_key", "round"], how="left")
       .join(plant, on=["match_key", "round"], how="left")
       .join(roster, on=["match_key", "round"], how="left")
       .join(dead, on=["match_key", "round"], how="left"))
T = T.with_columns(
    pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False),
    pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0),
    pl.col("n_T").fill_null(0), pl.col("n_CT").fill_null(0),
).with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_bomb"),
    pl.when(pl.col("defused")).then(pl.lit(3)).when(pl.col("exploded")).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("BO"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .otherwise(pl.lit("other")).alias("rclass"),
)


def passes(w):
    return ((pl.col(w) == 2) & (pl.col("CT_dead") >= pl.col("n_CT")) & (pl.col("n_CT") > 0)) | \
           ((pl.col(w) == 3) & (pl.col("T_dead") >= pl.col("n_T")) & (pl.col("n_T") > 0))


T = T.with_columns(
    passes("RE").alias("de_RE"),
    passes("SB").alias("de_SB"),
    pl.when(pl.col("RE") == 2).then(pl.lit(3)).otherwise(pl.lit(2)).alias("RE_flip"),
).with_columns(passes("RE_flip").alias("de_REflip"))

NB = T.filter(~pl.col("has_bomb") & pl.col("SB").is_not_null())
print("\n== non-bomb rounds with SB: death-check pass rate by build x rclass (n)")
print(NB.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("de_RE").mean().round(4).alias("pass_RE"),
    pl.col("de_SB").mean().round(4).alias("pass_SB"),
    pl.col("de_REflip").mean().round(4).alias("pass_RE_flipped"),
).sort(["build_num", "rclass"]))

BOR = T.filter(pl.col("has_bomb") & pl.col("SB").is_not_null())
print("\n== bomb rounds with SB: agreement with bomb outcome BO (n)")
print(BOR.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    (pl.col("RE") == pl.col("BO")).mean().round(4).alias("RE_agrees_BO"),
    (pl.col("SB") == pl.col("BO")).mean().round(4).alias("SB_agrees_BO"),
).sort(["build_num", "rclass"]))

print("\n== planter side (bomb_planted player_team_code) by build x rclass: counts of 2 (T), 3 (CT), other")
PL = T.filter(pl.col("planter_side").is_not_null())
print(PL.group_by(["build_num", "rclass"]).agg(
    (pl.col("planter_side") == 2).sum().alias("planter_T"),
    (pl.col("planter_side") == 3).sum().alias("planter_CT"),
    (~pl.col("planter_side").is_in([2, 3])).sum().alias("planter_other"),
    pl.len().alias("n_planted_rounds"),
).sort(["build_num", "rclass"]))

print("\n== roster size (n_T, n_CT) in non-bomb SB rounds, build 9961 by rclass: share with n_T<5 or n_CT<5")
R9 = NB.filter(pl.col("build_num") == 9961).with_columns(
    ((pl.col("n_T") < 5) | (pl.col("n_CT") < 5)).alias("short_roster"))
print(R9.group_by("rclass").agg(pl.len().alias("n"), pl.col("short_roster").mean().round(4).alias("short_share"),
                                 pl.col("n_roster").mean().round(2).alias("mean_roster")))

print("\n== win reason message for non-bomb rounds, build 9961, round 13 (count)")
print(NB.filter((pl.col("build_num") == 9961) & (pl.col("rclass") == "r13")).group_by(
    ["win_reason_message", "SB"]).len().sort("len", descending=True).head(10))

T.write_parquet(SCR / "three_source_round.parquet")
print("\nwrote", SCR / "three_source_round.parquet", T.height)
