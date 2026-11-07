"""verify_D2_3_07: exact counts used in the verdict.

  C1  round-9 rows, build 9961: bomb (B) and score side (S) agree with each other and disagree with recorded winner (R)
  C2  2024 round-states (all 2024 rounds with a recorded winner) and how many sit in build-9961 matches flagged at round 13
      (score truth S disagrees with R at 13), split into rounds 13/25/28 (label defect) and rounds 14+ (lead offset).
  C3  build-9961 matches with score truth at round 13 present: how many flagged.
  C4  E vs B agreement (elimination truth reliability, bomb rounds), by build.

Reads 2024 tome only. Column projection; one channel at a time. No files written.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_07_counts.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
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


hdr = pl.read_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*"))),
                      columns=["key", "build_num"]).rename({"key": MK})
re_raw = pl.scan_parquet(files("round_end")).select([MK, "round", "tick", "winner_team_code"]).collect()
re_ = re_raw.sort("tick").unique([MK, "round"], keep="first").rename({"tick": "t_end"})
print("2024 round_end (match,round) rows with a winner:", re_.height, " matches:", re_[MK].n_unique())

rs = (pl.scan_parquet(files("round_state")).select([MK, "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = rs.sort("tick").unique([MK, "round"], keep="first").select([MK, "round", "t_score", "ct_score", pl.col("tick").alias("t_fe")])
after = fe.select([MK, (pl.col("round") - 1).alias("round"), pl.col("t_score").alias("t_a"), pl.col("ct_score").alias("ct_a")])
st = fe.join(after, on=[MK, "round"], how="inner").with_columns(
    (pl.col("t_a") - pl.col("t_score")).alias("dT"), (pl.col("ct_a") - pl.col("ct_score")).alias("dCT"))
st = st.with_columns(
    pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
    .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("S"))

bs = pl.scan_parquet(files("bomb_state")).select([MK, "round", "tick", "event_type"]).collect()
bs = (bs.join(fe.select([MK, "round", "t_fe"]), on=[MK, "round"], how="left")
      .join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL)).alias("in_win")))
ob = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]) & pl.col("in_win"))
      .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
      .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                    .cast(pl.Int64).alias("B")))

base = (re_.select([MK, "round", "winner_team_code"])
        .join(st.select([MK, "round", "S"]), on=[MK, "round"], how="left")
        .join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="left")
        .join(hdr, on=MK, how="left"))

# ---------------------------------------------------------------- C1
r9 = base.filter((pl.col("round") == 9) & (pl.col("build_num") == 9961))
print("\n== C1. Round 9, build 9961")
print("rows:", r9.height, " with B:", r9.filter(pl.col("B").is_not_null()).height,
      " with S:", r9.filter(pl.col("S").is_not_null()).height)
print("B and R disagree (bomb-confirmed R error):", r9.filter(pl.col("B").is_not_null() & (pl.col("B") != pl.col("winner_team_code"))).height)
print("crosstab of (B vs R) and (S vs R) where both B and S exist:")
print(r9.filter(pl.col("B").is_not_null() & pl.col("S").is_not_null())
      .group_by([(pl.col("B") != pl.col("winner_team_code")).alias("B_ne_R"),
                 (pl.col("S") != pl.col("winner_team_code")).alias("S_ne_R"),
                 (pl.col("S") == pl.col("B")).alias("S_eq_B")]).len()
      .sort(["B_ne_R", "S_ne_R", "S_eq_B"]))
print("bomb-confirmed R errors (B=3 defused, R=2 T recorded, S=3):",
      r9.filter((pl.col("B") == 3) & (pl.col("winner_team_code") == 2) & (pl.col("S") == 3)).height,
      " (B=2, R=3, S=2):", r9.filter((pl.col("B") == 2) & (pl.col("winner_team_code") == 3) & (pl.col("S") == 2)).height)
print("score-only disagreements (S and R, no B):",
      r9.filter(pl.col("B").is_null() & pl.col("S").is_not_null() & (pl.col("S") != pl.col("winner_team_code"))).height,
      " of which S=2/R=3:", r9.filter(pl.col("B").is_null() & (pl.col("S") == 2) & (pl.col("winner_team_code") == 3)).height,
      " S=3/R=2:", r9.filter(pl.col("B").is_null() & (pl.col("S") == 3) & (pl.col("winner_team_code") == 2)).height)

# ---------------------------------------------------------------- C2 / C3
print("\n== C2/C3. Build-9961 flag at round 13 (S disagrees with R) and affected round-states")
f13 = (base.filter((pl.col("round") == 13) & (pl.col("build_num") == 9961) & pl.col("S").is_not_null())
       .select([MK, (pl.col("S") != pl.col("winner_team_code")).alias("flag13")]))
print("build-9961 matches with S at round 13:", f13.height, " flagged:", f13.filter("flag13").height)
allm9961 = hdr.filter(pl.col("build_num") == 9961).height
print("build-9961 matches in 2024 tome:", allm9961)
bf = base.join(f13, on=MK, how="inner").filter(pl.col("flag13"))
print("round-states in flagged build-9961 matches, total:", bf.height)
print("  of which rounds 13/25/28 (label defect, recorded winner wrong):",
      bf.filter(pl.col("round").is_in([13, 25, 28])).height)
print("  of which rounds 14+ other than 25/28 (cumulative lead offset from round 13):",
      bf.filter((pl.col("round") >= 14) & ~pl.col("round").is_in([25, 28])).height)
print("  of which rounds 1-12 (before the defect; lead not offset):", bf.filter(pl.col("round") <= 12).height)
print("total 2024 round rows with a recorded winner:", base.height)
print("share of all 2024 round rows that are label-wrong or lead-offset (flagged build 9961, round >= 13):",
      round(bf.filter(pl.col("round") >= 13).height / base.height, 4))

# ---------------------------------------------------------------- C4
print("\n== C4. Elimination truth reliability (E vs B where both exist), by build")
pdth = pl.scan_parquet(files("player_death")).select([MK, "round", "tick", "player_team_code"]).collect()
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
c4 = base.join(el.select([MK, "round", "E"]), on=[MK, "round"], how="left").filter(
    pl.col("B").is_not_null() & pl.col("E").is_not_null())
print(c4.group_by("build_num").agg(pl.len().alias("n_bomb_and_E"), (pl.col("E") == pl.col("B")).sum().alias("E_eq_B")))
print(c4.with_columns(pl.when(pl.col("round").is_in([13, 25, 28])).then(pl.lit("swap")).otherwise(pl.lit("other")).alias("kind"))
      .group_by(["build_num", "kind"]).agg(pl.len().alias("n"), (pl.col("E") == pl.col("B")).sum().alias("E_eq_B")))

print("\nend of count tables (no files written)")
