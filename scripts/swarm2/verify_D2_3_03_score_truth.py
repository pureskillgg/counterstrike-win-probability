"""verify_D2_3_03: bomb-free truth from freeze-end score deltas, to (a) test the build-level pattern
without bomb events and (b) cover build-9961 matches with no round-13 bomb outcome (days 07-11).

Truth S (score-based, no sx, no bomb):
  S[r] = 2 if t_score rose by 1 and ct_score unchanged between round_freeze_end of r and of r+1
       = 3 if ct_score rose by 1 and t_score unchanged
       else null
  ASSUMPTION A5: round_state t_score / ct_score are SIDE-attached (T-side and CT-side scores).
  A5 is tested, not assumed alone: it must agree with R (winner_team_code) at ~100% in non-swap rounds 2-12 and 14-24
  of both builds, and with the bomb truth B wherever both exist.

Also:
  - build-9961 round-9 flips (10 rows): do they carry a score-side flip?
  - round-13 bomb coverage by day for build 9961 vs 9842.
  - winner-derived cumulative lead vs freeze-end score lead, by build, round >= 14 (propagation check).

Reads 2024 tome only. Column projection. No files written.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_03_score_truth.py
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
TOL = 40  # ASSUMPTION A1 (bomb window, as in 01)


def files(ch):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def scan(ch, cols):
    return pl.scan_parquet(files(ch)).select(cols)


hdr = pl.read_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*"))),
                      columns=["key", "map_name", "build_num", "match_date"]).rename({"key": MK})
hdr = hdr.with_columns(pl.col(MK).str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

re_raw = scan("round_end", [MK, "round", "tick", "winner_team_code", "player_count"]).collect()
re_ = re_raw.sort("tick").unique([MK, "round"], keep="first").rename({"tick": "t_end"})

# ---------------------------------------------------------------- freeze-end scores (A5)
rs = (scan("round_state", [MK, "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = rs.sort("tick").unique([MK, "round"], keep="first").select([MK, "round", "t_score", "ct_score", pl.col("tick").alias("t_fe")])
print("freeze_end (match,round) rows:", fe.height, " matches:", fe[MK].n_unique())
after = fe.select([MK, (pl.col("round") - 1).alias("round"),
                   pl.col("t_score").alias("t_a"), pl.col("ct_score").alias("ct_a")])
st = fe.join(after, on=[MK, "round"], how="inner").with_columns(
    (pl.col("t_a") - pl.col("t_score")).alias("dT"), (pl.col("ct_a") - pl.col("ct_score")).alias("dCT"))
st = st.with_columns(
    pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
    .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("S"))
print("rounds with a consecutive freeze_end pair:", st.height, " with score-side S:", st.filter(pl.col("S").is_not_null()).height)
print("dT,dCT patterns:", st.group_by(["dT", "dCT"]).len().sort(["dT", "dCT"]).rows())

# ---------------------------------------------------------------- bomb truth B (as in 01)
bs = scan("bomb_state", [MK, "round", "tick", "event_type"]).collect()
bs = (bs.join(fe.select([MK, "round", "t_fe"]), on=[MK, "round"], how="left")
      .join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL)).alias("in_win")))
ob = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]) & pl.col("in_win"))
      .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
      .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                    .cast(pl.Int64).alias("B")))

base = (re_.select([MK, "round", "winner_team_code", "player_count"])
        .join(st.select([MK, "round", "S"]), on=[MK, "round"], how="left")
        .join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="left")
        .join(hdr.select([MK, "map_name", "build_num", "day"]), on=MK, how="left")
        .with_columns((pl.col("winner_team_code") != pl.col("S")).alias("flip_S"),
                      (pl.col("winner_team_code") != pl.col("B")).alias("flip_B")))
print("base rows:", base.height, " with build:", base.filter(pl.col("build_num").is_not_null()).height)

# ---------------------------------------------------------------- A5 validation
print("\n== V1. A5 check: score side S vs recorded winner R, by build and round (non-null S)")
v = base.filter(pl.col("S").is_not_null())
rr = pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25")) \
    .when(pl.col("round") == 28).then(pl.lit("r28")).when(pl.col("round") <= 12).then(pl.lit("r01_12")) \
    .when(pl.col("round") <= 24).then(pl.lit("r14_24")).otherwise(pl.lit("r26_27_29plus"))
print(v.with_columns(rr.alias("rbin")).group_by(["build_num", "rbin"])
      .agg(pl.len().alias("n"), pl.col("flip_S").sum().alias("flip_S")).sort(["build_num", "rbin"]))

print("\n== V2. S vs B where both exist, by build (A5 vs bomb truth)")
both = base.filter(pl.col("S").is_not_null() & pl.col("B").is_not_null())
print(both.group_by(["build_num"]).agg(pl.len().alias("n"), (pl.col("S") == pl.col("B")).sum().alias("S_eq_B")))
print(both.with_columns(rr.alias("rbin")).group_by(["build_num", "rbin"])
      .agg(pl.len().alias("n"), (pl.col("S") == pl.col("B")).sum().alias("S_eq_B")).sort(["build_num", "rbin"]))

# ---------------------------------------------------------------- round-13 flips using score truth, all matches
print("\n== V3. Round 13 flips using S (bomb-free), by build; coverage vs B")
r13 = base.filter(pl.col("round") == 13)
print(r13.group_by(["build_num"]).agg(
    pl.len().alias("rows"),
    pl.col("S").is_not_null().sum().alias("with_S"),
    pl.col("flip_S").sum().alias("flip_S"),
    pl.col("B").is_not_null().sum().alias("with_B"),
    pl.col("flip_B").sum().alias("flip_B")).sort("build_num"))
print("flip_S rate among rows with S, by build:")
print(r13.filter(pl.col("S").is_not_null()).group_by("build_num")
      .agg(pl.len().alias("n"), pl.col("flip_S").sum().alias("flipped"), pl.col("flip_S").mean().round(3).alias("rate")))

print("\n== V4. Build 9961 round-13 flips by S, by day (includes days 07-11 with no bomb outcome)")
print(r13.filter(pl.col("build_num") == 9961).group_by("day").agg(
    pl.len().alias("rows"), pl.col("S").is_not_null().sum().alias("with_S"),
    pl.col("flip_S").sum().alias("flip_S"), pl.col("B").is_not_null().sum().alias("with_B"))
      .sort("day").rows())

print("\n== V5. Bomb-outcome coverage by round and build (share of rounds with an in-window outcome)")
cov = (base.with_columns(pl.col("B").is_not_null().alias("hasB"))
       .group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("hasB").mean().round(3).alias("share"))
       .filter(pl.col("round") <= 30).sort(["build_num", "round"]))
print(cov.rows())

print("\n== V6. Build 9961, days 07-11: bomb outcome share by round (is round 13 special?)")
early = base.filter((pl.col("build_num") == 9961) & pl.col("day").is_in(["07", "08", "09", "10", "11"]))
print(early.group_by("round").agg(pl.len().alias("n"), pl.col("B").is_not_null().mean().round(3).alias("share_B"),
                                  pl.col("S").is_not_null().mean().round(3).alias("share_S"))
      .filter(pl.col("round") <= 30).sort("round").rows())

# ---------------------------------------------------------------- round-9 pattern under S
print("\n== V7. Round 9 flips under S and B, by build")
r9 = base.filter(pl.col("round") == 9)
print(r9.group_by("build_num").agg(pl.len().alias("n"), pl.col("S").is_not_null().sum().alias("with_S"),
                                   pl.col("flip_S").sum().alias("flip_S"),
                                   pl.col("B").is_not_null().sum().alias("with_B"),
                                   pl.col("flip_B").sum().alias("flip_B")))
print("round-9 rows where flip_S is true (by build, S, B):")
print(r9.filter(pl.col("flip_S") == True).select([MK, "winner_team_code", "S", "B", "build_num", "day"]).sort(MK))  # noqa: E712

# ---------------------------------------------------------------- score-lead propagation check
print("\n== V8. Propagation: winner-derived cumulative lead vs score lead at freeze_end, by build and round band")
# score lead at freeze_end of round r: ct_score - t_score (side-attached), i.e., cumulative CT-minus-T wins before r
lead = (fe.with_columns((pl.col("ct_score") - pl.col("t_score")).alias("lead_score")).select([MK, "round", "lead_score"]))
rw = re_.select([MK, "round", "winner_team_code"]).with_columns(
    pl.when(pl.col("winner_team_code") == 3).then(pl.lit(1)).otherwise(pl.lit(-1)).alias("step"))
rw = rw.sort([MK, "round"]).with_columns(
    pl.col("step").cum_sum().over(MK).alias("cum_R"))  # cum through round r (inclusive)
lead_R = rw.with_columns((pl.col("round") + 1).alias("round")).select([MK, "round", pl.col("cum_R").alias("lead_R_before")])
cmp = lead.join(lead_R, on=[MK, "round"], how="inner").join(hdr.select([MK, "build_num", "day"]), on=MK, how="left")
cmp = cmp.with_columns((pl.col("lead_R_before") - pl.col("lead_score")).alias("diff"))
print("rounds compared:", cmp.height)
print(cmp.with_columns(pl.when(pl.col("round") <= 13).then(pl.lit("r02_13"))
                       .when(pl.col("round") <= 24).then(pl.lit("r14_24")).otherwise(pl.lit("r25plus")).alias("band"))
      .group_by(["build_num", "band"]).agg(pl.len().alias("n"), (pl.col("diff") != 0).sum().alias("lead_mismatch"),
                                           pl.col("diff").abs().max().alias("max_abs_diff"))
      .sort(["build_num", "band"]))

print("\nend of score-truth tables (no files written)")
