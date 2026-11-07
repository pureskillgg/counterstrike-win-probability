"""verify_D2_3_05: rows with NO bomb outcome and NO elimination (time-out-like rounds).

Rule (CS2 round logic, no bomb plant and no side wiped): the round can only end by time, and time-out is a CT win.
So for rows with B null and E null, a T-side winner (2) is only possible if the death events are incomplete.
This script tests which source is right in those rows: R (recorded winner), S (score-delta side), and the
round_end win_reason_message. ASSUMPTION A7: a time-out round is a CT win (no bomb planted, no side wiped).

Also tests whether win_reason_message is consistent with R, and whether the round-9 and round-13 S-only flips
(verify_D2_3_04 Q2/Q3) are time-out rows.

Reads 2024 tome only. Column projection; one channel at a time. No files written.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_05_timeout_rows.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
MK = "match_key"
TOL = 40  # ASSUMPTION A1


def files(ch):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def scan(ch, cols):
    return pl.scan_parquet(files(ch)).select(cols)


hdr = pl.read_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*"))),
                      columns=["key", "map_name", "build_num"]).rename({"key": MK})
hdr = hdr.with_columns(pl.col(MK).str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

re_raw = scan("round_end", [MK, "round", "tick", "winner_team_code", "win_reason_code",
                            "win_reason_message", "player_count"]).collect()
re_ = re_raw.sort("tick").unique([MK, "round"], keep="first").rename({"tick": "t_end"})

rs = (scan("round_state", [MK, "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = rs.sort("tick").unique([MK, "round"], keep="first").select(
    [MK, "round", "t_score", "ct_score", pl.col("tick").alias("t_fe")])
after = fe.select([MK, (pl.col("round") - 1).alias("round"), pl.col("t_score").alias("t_a"),
                   pl.col("ct_score").alias("ct_a")])
st = fe.join(after, on=[MK, "round"], how="inner").with_columns(
    (pl.col("t_a") - pl.col("t_score")).alias("dT"), (pl.col("ct_a") - pl.col("ct_score")).alias("dCT"))
st = st.with_columns(
    pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
    .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("S"))

bs = scan("bomb_state", [MK, "round", "tick", "event_type"]).collect()
bs = (bs.join(fe.select([MK, "round", "t_fe"]), on=[MK, "round"], how="left")
      .join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL)).alias("in_win")))
ob = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]) & pl.col("in_win"))
      .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
      .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                    .cast(pl.Int64).alias("B")))

pdth = scan("player_death", [MK, "round", "tick", "player_team_code"]).collect()
pdth = (pdth.join(fe.select([MK, "round", "t_fe"]), on=[MK, "round"], how="left")
        .join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
        .filter(pl.col("t_fe").is_not_null() & (pl.col("tick") >= pl.col("t_fe"))
                & (pl.col("tick") <= pl.col("t_end") + TOL) & pl.col("player_team_code").is_in([2, 3]))
        .sort("tick"))
pdth = pdth.with_columns(pl.col("tick").cum_count().over([MK, "round", "player_team_code"]).alias("nth"))
fifth = pdth.filter(pl.col("nth") == 5).select([MK, "round", "player_team_code", pl.col("tick").alias("t5")])
t_out = fifth.filter(pl.col("player_team_code") == 2).select([MK, "round", pl.col("t5").alias("t5_T")])
c_out = fifth.filter(pl.col("player_team_code") == 3).select([MK, "round", pl.col("t5").alias("t5_CT")])
el = (re_.select([MK, "round"]).join(t_out, on=[MK, "round"], how="left").join(c_out, on=[MK, "round"], how="left"))
el = el.with_columns(
    pl.when(pl.col("t5_T").is_not_null() & pl.col("t5_CT").is_not_null()).then(
        pl.when(pl.col("t5_T") < pl.col("t5_CT")).then(pl.lit(3)).otherwise(pl.lit(2)))
    .when(pl.col("t5_T").is_not_null()).then(pl.lit(3))
    .when(pl.col("t5_CT").is_not_null()).then(pl.lit(2))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("E"))

base = (re_.select([MK, "round", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"])
        .join(st.select([MK, "round", "S"]), on=[MK, "round"], how="left")
        .join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="left")
        .join(el.select([MK, "round", "E"]), on=[MK, "round"], how="left")
        .join(hdr, on=MK, how="left"))
base = base.with_columns(
    pl.when(pl.col("round") == 9).then(pl.lit("r09"))
    .when(pl.col("round") == 13).then(pl.lit("r13"))
    .when(pl.col("round") == 25).then(pl.lit("r25"))
    .when(pl.col("round") == 28).then(pl.lit("r28"))
    .when(pl.col("round") <= 12).then(pl.lit("r01_12"))
    .when(pl.col("round") <= 24).then(pl.lit("r14_24"))
    .otherwise(pl.lit("r26_27_29p")).alias("rbin"))

tc = base.filter(pl.col("B").is_null() & pl.col("E").is_null())
print("rows with no bomb and no elimination (time-out-like):", tc.height)
print("by build and rbin:")
print(tc.group_by(["build_num", "rbin"]).agg(pl.len().alias("n")).sort(["build_num", "rbin"]))

print("\n== T1. win_reason_message by recorded winner, time-out-like rows, by build")
print(tc.group_by(["build_num", "winner_team_code", "win_reason_message"]).len()
      .sort(["build_num", "winner_team_code", "win_reason_message"]))

print("\n== T2. Time-out-like rows: recorded winner R vs score side S, by build and rbin")
tc2 = tc.with_columns((pl.col("S") == pl.col("winner_team_code")).alias("S_eq_R"))
print(tc2.group_by(["build_num", "rbin", "S"]).agg(pl.len().alias("n"),
                                                    pl.col("winner_team_code").eq(3).sum().alias("R_is_CT"),
                                                    pl.col("S_eq_R").sum().alias("S_eq_R"))
      .sort(["build_num", "rbin", "S"]))

print("\n== T3. Time-out-like rows: R distribution (2 = T, 3 = CT), by build and rbin")
print(tc.group_by(["build_num", "rbin", "winner_team_code"]).len().sort(["build_num", "rbin", "winner_team_code"]))

print("\n== T4. Rows where S says T won but no bomb and no elimination: win_reason and R (all builds)")
s_t = tc.filter(pl.col("S") == 2)
print(s_t.group_by(["build_num", "rbin", "winner_team_code", "win_reason_message"]).len()
      .sort(["build_num", "rbin", "winner_team_code", "win_reason_message"]))

print("\n== T5. Round-level check of the claimed mechanism: does S disagree with R anywhere outside time-outs?")
nt = base.filter(~(pl.col("B").is_null() & pl.col("E").is_null()))
print("rows with B or E present:", nt.height,
      " S disagrees with R:", nt.filter(pl.col("S").is_not_null() & (pl.col("S") != pl.col("winner_team_code"))).height)
print(nt.filter(pl.col("S").is_not_null() & (pl.col("S") != pl.col("winner_team_code")))
      .group_by(["build_num", "rbin"]).len().sort(["build_num", "rbin"]))

print("\nend of time-out tables (no files written)")
