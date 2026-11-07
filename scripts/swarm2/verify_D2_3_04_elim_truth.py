"""verify_D2_3_04: third, bomb-free and score-free truth for rounds without a bomb outcome: elimination.

E (elimination truth), side-attached via player_death.player_team_code (2 = T, 3 = CT), deaths inside the round window:
  the side whose 5th death comes first LOSES the round; E = winning side. Null if neither side reaches 5 deaths
  inside the window (time-out, or bomb round).
  ASSUMPTION A6: player_death.player_team_code is the victim's side at the time of death.
  ASSUMPTION A1 as before: window [round_freeze_end, round_end + 40 ticks].

Compared against:
  R = recorded winner_team_code
  S = score-delta side (verify_D2_3_03, A5)
  B = bomb-outcome side (verify_D2_3_01)

Questions answered:
  Q1  Does E agree with R where R is trusted (rounds 2-12, 14-24, both builds, non-bomb rounds)? (validates A6)
  Q2  In build 9961, non-bomb round 9 and round 13 rows: R vs E vs S. Is the round-9 score flip (292 rows) real?
  Q3  Build 9961 round-13 rows where S says no flip: does E agree with R (true no-flip) or with S (S-only noise)?
  Q4  Match-level pattern at 13/25/28 using all truths, crossed with build_num.

Reads 2024 tome only. Column projection; one channel at a time. No files written.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_04_elim_truth.py
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


def rbin(c="round"):
    r = pl.col(c)
    return (pl.when(r <= 12).then(pl.lit("r01_12")).when(r == 13).then(pl.lit("r13"))
            .when(r <= 24).then(pl.lit("r14_24")).when(r == 25).then(pl.lit("r25"))
            .when(r <= 27).then(pl.lit("r26_27")).when(r == 28).then(pl.lit("r28"))
            .otherwise(pl.lit("r29plus")).alias("rbin"))


hdr = pl.read_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*"))),
                      columns=["key", "map_name", "build_num", "match_date"]).rename({"key": MK})
hdr = hdr.with_columns(pl.col(MK).str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

re_raw = scan("round_end", [MK, "round", "tick", "winner_team_code", "player_count"]).collect()
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

# ---------------------------------------------------------------- elimination truth E
pdth = scan("player_death", [MK, "round", "tick", "player_team_code"]).collect()
print("player_death rows:", pdth.height, " player_team_code values:",
      pdth["player_team_code"].value_counts().sort("player_team_code").rows())
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
    .when(pl.col("t5_T").is_not_null()).then(pl.lit(3))   # T wiped first -> CT wins
    .when(pl.col("t5_CT").is_not_null()).then(pl.lit(2))  # CT wiped first -> T wins
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("E"))
print("rounds with E (5th death of one side in window):", el.filter(pl.col("E").is_not_null()).height)

# ---------------------------------------------------------------- base
base = (re_.select([MK, "round", "winner_team_code", "player_count"])
        .join(st.select([MK, "round", "S", "dT", "dCT"]), on=[MK, "round"], how="left")
        .join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="left")
        .join(el.select([MK, "round", "E"]), on=[MK, "round"], how="left")
        .join(hdr.select([MK, "map_name", "build_num", "day", "match_date"]), on=MK, how="left")
        .with_columns(rbin("round"), (pl.col("winner_team_code") != pl.col("S")).alias("fS"),
                      (pl.col("winner_team_code") != pl.col("B")).alias("fB"),
                      (pl.col("winner_team_code") != pl.col("E")).alias("fE")))
print("base rows:", base.height, " with E:", base.filter(pl.col("E").is_not_null()).height,
      " with B:", base.filter(pl.col("B").is_not_null()).height,
      " with S:", base.filter(pl.col("S").is_not_null()).height)

# ---------------------------------------------------------------- Q1: validate E on rounds where R is trusted
print("\n== Q1. E vs R where R is trusted (non-bomb rounds: B null, E non-null)")
nb = base.filter(pl.col("B").is_null() & pl.col("E").is_not_null())
print(nb.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("fE").sum().alias("E_ne_R"))
      .sort(["build_num", "rbin"]))
print("E vs B where both exist (bomb rounds):")
both = base.filter(pl.col("B").is_not_null() & pl.col("E").is_not_null())
print("  n =", both.height, " E==B:", both.filter(pl.col("E") == pl.col("B")).height)

# ---------------------------------------------------------------- Q2: round 9 and round 13, non-bomb rows
print("\n== Q2. Non-bomb rows, round 9 and 13: R vs E vs S, by build")
q2 = base.filter(pl.col("B").is_null() & pl.col("round").is_in([9, 13]))
print(q2.group_by(["round", "build_num"]).agg(
    pl.len().alias("rows"),
    pl.col("E").is_not_null().sum().alias("with_E"), pl.col("fE").sum().alias("E_ne_R"),
    pl.col("S").is_not_null().sum().alias("with_S"), pl.col("fS").sum().alias("S_ne_R"),
    (pl.col("fS") & pl.col("fE")).sum().alias("S_and_E_ne_R"),
    (pl.col("fS") & pl.col("E").is_not_null() & (pl.col("E") == pl.col("S"))).sum().alias("S_eq_E_when_flipped")
).sort(["round", "build_num"]))

print("\nround 9, build 9961, non-bomb: crosstab of (S disagrees with R) x (E disagrees with R)")
print(base.filter((pl.col("round") == 9) & (pl.col("build_num") == 9961) & pl.col("B").is_null())
      .group_by(["fS", "fE"]).len().sort(["fS", "fE"]))
print("round 9, build 9961, non-bomb: where S disagrees with R, what E says (E as side code vs R, S)")
print(base.filter((pl.col("round") == 9) & (pl.col("build_num") == 9961) & pl.col("B").is_null()
                  & pl.col("fS") & pl.col("E").is_not_null())
      .group_by(["winner_team_code", "S", "E"]).len().sort(["winner_team_code", "S", "E"]))

# ---------------------------------------------------------------- Q3: round 13 in build 9961, S says no flip
print("\n== Q3. Build 9961 round 13: crosstab of R vs B, S, E (rows with each truth available)")
r13 = base.filter((pl.col("round") == 13) & (pl.col("build_num") == 9961))
print(r13.group_by(["fS", "fB", "fE"]).len().sort(["fS", "fB", "fE"]))
print("rows with S but no B and no E:", r13.filter(pl.col("S").is_not_null() & pl.col("B").is_null()
                                                  & pl.col("E").is_null()).height)
print("rows with no truth at all (S, B, E all null):", r13.filter(pl.col("S").is_null() & pl.col("B").is_null()
                                                             & pl.col("E").is_null()).height)

print("\n== Q3b. Build 9842 round 13 crosstab (control)")
r13c = base.filter((pl.col("round") == 13) & (pl.col("build_num") == 9842))
print(r13c.group_by(["fS", "fB", "fE"]).len().sort(["fS", "fB", "fE"]))

# ---------------------------------------------------------------- Q4: match-level at 13/25/28 with all truths
print("\n== Q4. Match-level flips at 13/25/28 with S (bomb-free) truth, crossed with build_num")
piv = (base.filter(pl.col("round").is_in([13, 25, 28]))
       .select([MK, pl.col("round").cast(pl.Utf8), "fS", "build_num"])
       .pivot(on="round", index=[MK, "build_num"], values="fS", aggregate_function="first"))
print("matches with S at 13, 25, 28 all present:", piv.drop_nulls(["13", "25", "28"]).height)
print(piv.drop_nulls(["13", "25", "28"]).group_by(["build_num", "13", "25", "28"]).len()
      .sort(["build_num", "13", "25", "28"]))
print("matches with S at 13 present, by build and flip at 13:")
print(piv.drop_nulls(["13"]).group_by(["build_num", "13"]).len().sort(["build_num", "13"]))
print("matches with S at 25 present, by build and flip at 25:")
print(piv.drop_nulls(["25"]).group_by(["build_num", "25"]).len().sort(["build_num", "25"]))
print("matches with S at 28 present, by build and flip at 28:")
print(piv.drop_nulls(["28"]).group_by(["build_num", "28"]).len().sort(["build_num", "28"]))

# ---------------------------------------------------------------- Q5: round-level summary with the strongest truth available
print("\n== Q5. Round-level flip rate using the strongest available truth (B > E > S), by build and round bin")
base = base.with_columns(pl.coalesce(["B", "E", "S"]).alias("T_best"))
base = base.with_columns((pl.col("winner_team_code") != pl.col("T_best")).alias("fT"))
q5 = base.filter(pl.col("T_best").is_not_null())
print(q5.group_by(["build_num", "rbin"]).agg(pl.len().alias("n"), pl.col("fT").sum().alias("flipped"),
                                             pl.col("fT").mean().round(4).alias("rate")).sort(["build_num", "rbin"]))
print("\nround 9 and any other rounds with flips (T_best), build 9961 and 9842:")
print(q5.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("fT").sum().alias("flipped"))
      .filter(pl.col("flipped") > 0).sort(["build_num", "round"]).rows())

print("\nend of elimination-truth tables (no files written)")
