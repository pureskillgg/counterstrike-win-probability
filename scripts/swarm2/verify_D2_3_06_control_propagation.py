"""verify_D2_3_06: (a) control: build_num in the 2026 tome, so build can be tested as the discriminator;
(b) propagation: in 2024, does a flipped round 13 shift the cumulative winner-derived lead for rounds 14+?
(c) one traced match: recorded winner R, score side S, bomb B, elimination E for rounds 11-15.

Score side S as in verify_D2_3_03 (A5). Cumulative lead uses +1 for a CT win, -1 for a T win (side-attached),
so cum_R(r) - cum_S(r) is the offset that a recorded-winner-derived lead carries relative to the score truth.
Reads tomes only. Column projection. No files written.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_06_control_propagation.py
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
MK = "match_key"
TOL = 40  # ASSUMPTION A1


def tome_files(tome, ch):
    fd = sorted(glob.glob(str(ROOT / "tomes" / tome / "tome" / "csds" / f"basic_{ch}.*")))
    assert len(fd) == 1, (tome, ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


# ---------------------------------------------------------------- (a) build_num control in 2026
h26 = pl.read_parquet(sorted(glob.glob(str(ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds" / "header" / "dataframe_*"))),
                      columns=["match_id", "build_num", "match_date", "map_name"])
print("== A. 2026 tome header: build_num counts (control for build as discriminator)")
print(h26.group_by("build_num").len().sort("build_num").rows())
print("2026 matches:", h26.height, " date range:", h26["match_date"].min(), "to", h26["match_date"].max())

# ---------------------------------------------------------------- 2024 base
BASE24 = "2024-02-g"
hdr = pl.read_parquet(sorted(glob.glob(str(ROOT / "tomes" / BASE24 / "tome" / "csds" / "header" / "dataframe_*"))),
                      columns=["key", "build_num", "map_name"]).rename({"key": MK})
re_raw = pl.scan_parquet(tome_files(BASE24, "round_end")).select([MK, "round", "tick", "winner_team_code"]).collect()
re_ = re_raw.sort("tick").unique([MK, "round"], keep="first").rename({"tick": "t_end"})
rs = (pl.scan_parquet(tome_files(BASE24, "round_state")).select([MK, "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = rs.sort("tick").unique([MK, "round"], keep="first").select([MK, "round", "t_score", "ct_score", pl.col("tick").alias("t_fe")])
after = fe.select([MK, (pl.col("round") - 1).alias("round"), pl.col("t_score").alias("t_a"), pl.col("ct_score").alias("ct_a")])
st = fe.join(after, on=[MK, "round"], how="inner").with_columns(
    (pl.col("t_a") - pl.col("t_score")).alias("dT"), (pl.col("ct_a") - pl.col("ct_score")).alias("dCT"))
st = st.with_columns(
    pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
    .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
    .otherwise(pl.lit(None)).cast(pl.Int64).alias("S"))
bs = pl.scan_parquet(tome_files(BASE24, "bomb_state")).select([MK, "round", "tick", "event_type"]).collect()
bs = (bs.join(fe.select([MK, "round", "t_fe"]), on=[MK, "round"], how="left")
      .join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL)).alias("in_win")))
ob = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]) & pl.col("in_win"))
      .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
      .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                    .cast(pl.Int64).alias("B")))

base = (re_.select([MK, "round", "winner_team_code", "t_end"])
        .join(st.select([MK, "round", "S"]), on=[MK, "round"], how="left")
        .join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="left")
        .join(hdr, on=MK, how="left")
        .sort([MK, "round"]))
base = base.with_columns(
    pl.when(pl.col("winner_team_code") == 3).then(pl.lit(1)).otherwise(pl.lit(-1)).alias("stepR"),
    pl.when(pl.col("S") == 3).then(pl.lit(1)).when(pl.col("S") == 2).then(pl.lit(-1)).otherwise(pl.lit(0)).alias("stepS"),
    pl.col("S").is_null().cast(pl.Int64).alias("nullS"))
base = base.with_columns(
    (pl.col("stepR").cum_sum().over(MK) - pl.col("stepR")).alias("cumR_before"),
    (pl.col("stepS").cum_sum().over(MK) - pl.col("stepS")).alias("cumS_before"),
    (pl.col("nullS").cum_sum().over(MK) - pl.col("nullS")).alias("nullS_before"))
# flag: S disagrees with R at round 13 (score truth), per match
f13 = (base.filter((pl.col("round") == 13) & pl.col("S").is_not_null())
       .select([MK, (pl.col("S") != pl.col("winner_team_code")).alias("flag13")]))
base = base.join(f13, on=MK, how="left").with_columns(pl.col("flag13").fill_null(False).alias("flag13"))
base = base.with_columns(pl.when(pl.col("round") == 13).then(pl.lit(0)).otherwise(pl.lit(0)).alias("_z"))

print("\n== B. Propagation: cumR_before - cumS_before (offset carried by recorded winners), 2024 tome")
print("   only matches with S known for every earlier round; flag13 = score side disagrees with R at round 13")
for r in [14, 20, 24, 27]:
    sub = base.filter((pl.col("round") == r) & (pl.col("nullS_before") == 0))
    sub = sub.with_columns((pl.col("cumR_before") - pl.col("cumS_before")).alias("offset"))
    print(f"-- round {r}")
    print(sub.group_by(["build_num", "flag13"]).agg(
        pl.len().alias("n"),
        (pl.col("offset") == 0).sum().alias("offset0"),
        (pl.col("offset").abs() == 2).sum().alias("offset_pm2"),
        pl.col("offset").abs().max().alias("max_abs")).sort(["build_num", "flag13"]).rows())

# ---------------------------------------------------------------- (c) trace one 9961 flagged match
print("\n== C. Trace: one build-9961 match flagged at round 13 (rounds 11-15)")
ex = (base.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("flag13")
                  & pl.col("B").is_not_null()).select(MK).head(1).item())
print("match:", ex)
tr = base.filter((pl.col("match_key") == ex) & pl.col("round").is_between(11, 15)).select(
    ["round", "winner_team_code", "S", "B", "cumR_before", "cumS_before"]).sort("round")
print(tr)
print("(winner_team_code and S, B use side codes: 2 = T, 3 = CT)")

# ---------------------------------------------------------------- (d) same match, rounds 25/28 and beyond
print("\n== D. Cross-check: does the flag-13 match also carry flips at 25 and 28 (score truth)?")
tr2 = base.filter((pl.col("match_key") == ex) & pl.col("round").is_in([24, 25, 26, 27, 28, 29, 30])).select(
    ["round", "winner_team_code", "S", "B"]).sort("round")
print(tr2)

print("\nend of control and propagation tables (no files written)")
