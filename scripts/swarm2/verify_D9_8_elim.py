"""verify D9-8, part 3: elimination check on NON-bomb rounds (2024 tome, read-only).

Question: is round_end winner_team_code wrong on round-13 rows that have no bomb outcome, or only on
bomb rows? Truth source here: player_death victims vs player_info roster per round (no bomb events,
no round_state scores, no round_end winner).

  elim_T_all_dead : every T player on the roster died in the round (distinct victims)
  elim_CT_all_dead: every CT player on the roster died in the round
A round ending by elimination must have exactly one of these true with the other false:
  winner 3 (CT) by elimination  -> elim_T_all_dead True,  elim_CT_all_dead False
  winner 2 (T)  by elimination  -> elim_CT_all_dead True, elim_T_all_dead False
Rounds used: round_end message is generic (#SFUI_Notice_CTs_Win or #SFUI_Notice_Terrorists_Win) and no
bomb_defused / bomb_exploded row in the round. Generic messages are used only to pick elimination-type
rounds; the check itself uses death and roster data.

ASSUMPTIONS:
  E1  player_death.player_team_code is the VICTIM's team code (2 = T, 3 = CT); calibrated on build 9842
      (the build with 0 bomb mismatches) before use on 9961
  E2  roster = distinct player_id per (match, round) with team_code 2 or 3 in player_info
  E3  a round with no roster (nT or nCT = 0) is excluded
Output is printed; no files are written under E:.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
GEN_T = "#SFUI_Notice_Terrorists_Win"
GEN_CT = "#SFUI_Notice_CTs_Win"


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


def sec(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


H = lz("header", ["key", "build_num"]).collect().rename({"key": "match_key"}).with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE = lz("round_end", ["match_key", "round", "winner_team_code", "win_reason_message", "player_count"]).collect()
RE = RE.unique(subset=["match_key", "round"], keep="first")
BO = lz("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).select(["match_key", "round"]).unique().collect()
PI = lz("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
DE = lz("player_death", ["match_key", "round", "player_id", "player_team_code"]).collect()

roster = (PI.filter(pl.col("team_code").is_in([2, 3])).unique(subset=["match_key", "round", "player_id"])
          .group_by(["match_key", "round"]).agg(
              (pl.col("team_code") == 2).sum().alias("nT"),
              (pl.col("team_code") == 3).sum().alias("nCT")))
victims = (DE.filter(pl.col("player_team_code").is_in([2, 3]))
           .unique(subset=["match_key", "round", "player_id", "player_team_code"])
           .group_by(["match_key", "round"]).agg(
               (pl.col("player_team_code") == 2).sum().alias("dT"),
               (pl.col("player_team_code") == 3).sum().alias("dCT")))

J = (RE.join(H.select(["match_key", "build_num", "day"]), on="match_key", how="left")
       .join(roster, on=["match_key", "round"], how="left")
       .join(victims, on=["match_key", "round"], how="left")
       .join(BO.with_columns(pl.lit(True).alias("has_bomb")), on=["match_key", "round"], how="left")
       .with_columns(
           pl.col("dT").fill_null(0), pl.col("dCT").fill_null(0), pl.col("has_bomb").fill_null(False),
           pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
             .otherwise(pl.lit("other")).alias("rbin"),
       ))
J = J.with_columns(
    (pl.col("dT") >= pl.col("nT")).alias("elim_T_all_dead"),
    (pl.col("dCT") >= pl.col("nCT")).alias("elim_CT_all_dead"),
    pl.col("win_reason_message").is_in([GEN_T, GEN_CT]).alias("generic"),
)
J = J.filter(pl.col("nT").is_not_null() & pl.col("nCT").is_not_null() & (pl.col("nT") > 0) & (pl.col("nCT") > 0))
print("rounds with roster (E3 applied):", J.height)

# consistency of round_end winner with elimination, on generic, non-bomb rounds
E = J.filter(pl.col("generic") & (~pl.col("has_bomb")))
E = E.with_columns(
    pl.when(pl.col("winner_team_code") == 3)
      .then(pl.col("elim_T_all_dead") & (~pl.col("elim_CT_all_dead")))
      .otherwise(pl.col("elim_CT_all_dead") & (~pl.col("elim_T_all_dead")))
      .alias("re_consistent_with_elim"),
    pl.when(pl.col("elim_T_all_dead") & (~pl.col("elim_CT_all_dead"))).then(pl.lit(3))
      .when(pl.col("elim_CT_all_dead") & (~pl.col("elim_T_all_dead"))).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("elim_winner"),
)
sec("E-A calibration: round_end winner vs elimination, generic non-bomb rounds, build 9842 (expect ~100%)")
cal = E.filter(pl.col("build_num") == 9842)
print("rows", cal.height, "| consistent", int(cal["re_consistent_with_elim"].sum()),
      "| rate", round(float(cal["re_consistent_with_elim"].mean()), 4))
print(cal.group_by(["rbin", "winner_team_code", "win_reason_message", "elim_winner"]).len()
      .sort(["rbin", "winner_team_code", "win_reason_message", "elim_winner"]))

sec("E-B build 9961 and 9842, non-bomb generic rounds: round_end vs elimination by round bin")
print(E.group_by(["build_num", "rbin"]).agg(
    pl.len().alias("n"),
    pl.col("re_consistent_with_elim").sum().alias("re_consistent"),
    pl.col("re_consistent_with_elim").mean().alias("re_rate"),
).sort(["build_num", "rbin"]))

sec("E-C build 9961 round 13 non-bomb generic rounds: round_end winner vs elimination winner")
e13 = E.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13))
print(e13.group_by(["winner_team_code", "win_reason_message", "elim_winner"]).len()
      .sort(["winner_team_code", "win_reason_message", "elim_winner"]))

sec("E-D build 9961 round 13 non-bomb rounds by day: round_end consistency (all messages, no bomb)")
allnb = J.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & (~pl.col("has_bomb")))
print("(all non-bomb rounds with roster, incl. non-generic messages, ~elimination only where a side is fully dead)")
print(allnb.group_by(["day", "winner_team_code"]).agg(
    pl.len().alias("n"),
    (pl.col("elim_T_all_dead") & ~pl.col("elim_CT_all_dead")).sum().alias("elim_T_only"),
    (pl.col("elim_CT_all_dead") & ~pl.col("elim_T_all_dead")).sum().alias("elim_CT_only"),
).sort(["day", "winner_team_code"]))

sec("E-E generic messages in build 9961 r13 non-bomb rounds vs elimination (2x2)")
print(e13.group_by(["winner_team_code", "elim_winner"]).len().sort(["winner_team_code", "elim_winner"]))

print("\nDONE")
