"""D9 probe 5: is a simple winner-code flip (2<->3) on build 9961 swap rounds validated by deaths?

For each non-bomb round, the as-recorded winner w and the flipped winner w' = 5 - w are scored with the
elimination rule: winner side = 2 (T) needs all CT dead (CT_dead >= n_CT); winner side = 3 (CT) needs
all T dead (T_dead >= n_T). Rounds with a Terrorists/CTs win message and no bomb outcome, and rounds
without any message restriction (any non-bomb round) are both reported.
Also counts how many bomb-outcome rounds would change label under the flip and whether the 7 build-9961
round-13 matches are rounds that the flip would break.

ASSUMPTIONS (ASSUMPTION): same as D9_tests4.py (victim side from player_death; n_T and n_CT from player_info).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D9")
R = pl.read_parquet(SCR / "D9_round.parquet")
M = pl.read_parquet(SCR / "D9_match.parquet")


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, want):
    fd = BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    return lf.select([c for c in want if c in have]).collect()


DE = scan("player_death", ["match_key", "round", "player_id", "player_team_code"])
DEu = DE.unique(subset=["match_key", "round", "player_id"], keep="first")
dead = (DEu.group_by(["match_key", "round"])
        .agg((pl.col("player_team_code") == 2).sum().alias("T_dead"),
             (pl.col("player_team_code") == 3).sum().alias("CT_dead")))
BS = scan("bomb_state", ["match_key", "round", "event_type"])
bomb_rounds = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .select(["match_key", "round"]).unique().with_columns(pl.lit(True).alias("bomb_any")))

T = (R.join(M.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(dead, on=["match_key", "round"], how="left")
       .join(bomb_rounds, on=["match_key", "round"], how="left")
       .with_columns(pl.col("bomb_any").fill_null(False),
                     pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
       .filter(pl.col("n_CT").is_not_null()))
T = T.with_columns(
    pl.when(pl.col("round") == 13).then(pl.lit("r13"))
      .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
      .otherwise(pl.lit("other")).alias("rclass"),
    (5 - pl.col("winner_team_code")).alias("winner_flipped"))


def ok(w_col):
    return ((pl.col(w_col) == 2) & (pl.col("CT_dead") >= pl.col("n_CT"))) | \
           ((pl.col(w_col) == 3) & (pl.col("T_dead") >= pl.col("n_T")))


T = T.with_columns(ok("winner_team_code").alias("ok_recorded"), ok("winner_flipped").alias("ok_flipped"))
NB = T.filter(~pl.col("bomb_any"))
print("== non-bomb rounds: rate consistent with deaths, recorded vs flipped winner")
print(NB.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("ok_recorded").mean().round(4).alias("rate_recorded"),
    pl.col("ok_flipped").mean().round(4).alias("rate_flipped")).sort(["build_num", "rclass"]))

print("\n== non-bomb 'Terrorists_Win' or 'CTs_Win' rounds only (no bomb)")
NBm = NB.filter(pl.col("win_reason_message").is_in(["#SFUI_Notice_Terrorists_Win", "#SFUI_Notice_CTs_Win"]))
print(NBm.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("ok_recorded").mean().round(4).alias("rate_recorded"),
    pl.col("ok_flipped").mean().round(4).alias("rate_flipped")).sort(["build_num", "rclass"]))

print("\n== bomb-outcome rounds: label change under flip (build 9961, pc20; by round class)")
BO = T.filter(pl.col("bomb_any") & (pl.col("player_count") == 20)).with_columns(
    pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
BO = BO.with_columns(
    (pl.col("bomb_side") == pl.col("winner_team_code")).alias("agree_recorded"),
    (pl.col("bomb_side") == pl.col("winner_flipped")).alias("agree_flipped"))
print(BO.group_by(["build_num", "rclass", "agree_recorded", "agree_flipped"]).len()
      .sort(["build_num", "rclass", "agree_recorded", "agree_flipped"]))

print("\n== the 7 build-9961 round-13 pc20 bomb rounds that already agree: deaths check and flip effect")
seven = BO.filter((pl.col("build_num") == 9961) & (pl.col("rclass") == "r13") & pl.col("agree_recorded"))
print(seven.select(["match_key", "winner_team_code", "win_reason_message", "bomb_side", "T_dead", "CT_dead",
                    "n_T", "n_CT", "ok_recorded", "ok_flipped"]))
