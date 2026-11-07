"""Verifier for D5-8, class pass: two classes of regime-A matches (round-9 vs round-13 inversions) and match format.

  C1. format fields (header number_of_points, round_start frag_limit/time_limit) by regime and by class.
  C2. boundary classes NS/SW at boundaries 8 and 12 (recorded winners) for regime-A late matches, split by
      round-9 step disagreement (class Y) and round-13 step disagreement (class X).
  C3. header test with per-match side convention: halftime at 12 (class X) or at 8 (class Y), flips at the
      first post-swap round only.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_class.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

B = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
SUF = "2023-12-28,2024-02-25"
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(B / ("header" if ch == "header" else f"basic_{ch}.{SUF}") / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed").collect()


R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})
H = rd("header", ["key", "number_of_points", "match_type", "tick_rate", "match_date"]).rename({"key": "match_key"})
RSTART = rd("round_start", ["match_key", "round", "time_limit", "frag_limit", "objective"])
rs1 = RSTART.sort(["match_key", "round"]).unique("match_key", keep="first", maintain_order=True).select(
    ["match_key", "time_limit", "frag_limit", "objective"])

M = R.select(["match_key", "regA", "day"]).unique("match_key").join(H, on="match_key", how="left").join(
    rs1, on="match_key", how="left")
M = M.with_columns(grp=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late")).when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
print("C1. format fields by regime (late and early):")
print(M.filter(pl.col("grp") != "other").group_by(["regA", "grp", "number_of_points", "frag_limit"]).len()
      .sort(["regA", "grp", "len"], descending=[False, False, True]).head(30))
print(M.filter(pl.col("grp") != "other").group_by(["regA", "grp", "match_type"]).len().sort(["regA", "grp"]))

# class Y/X membership for regime-A late matches
RA = R.filter(pl.col("regA") & pl.col("day").is_in(LATE))
s9 = RA.filter(pl.col("round") == 9).select(["match_key", ((pl.col("step_side").is_not_null()) & (pl.col("step_side") != pl.col("winner"))).alias("Y")])
s13 = RA.filter(pl.col("round") == 13).select(["match_key", ((pl.col("step_side").is_not_null()) & (pl.col("step_side") != pl.col("winner"))).alias("X")])
CL = RA.select("match_key").unique().join(s9, on="match_key", how="left").join(s13, on="match_key", how="left").with_columns(
    pl.col("Y").fill_null(False), pl.col("X").fill_null(False))
CL = CL.with_columns(cls=pl.when(pl.col("X")).then(pl.lit("X:r13")).when(pl.col("Y")).then(pl.lit("Y:r9")).otherwise(pl.lit("none")))
print("\nC1b. regime-A late matches by inversion class:")
print(CL.group_by("cls").len().sort("cls"))

# boundary classes at 8 and 12 with recorded winners (both classes)
FE = R.select(["match_key", "round", "t_fe", "ct_fe", "winner", "regA", "day"])
N = FE.with_columns(nt=pl.col("t_fe").shift(-1).over("match_key"), nct=pl.col("ct_fe").shift(-1).over("match_key"),
                    nr=pl.col("round").shift(-1).over("match_key"))
N = N.filter((pl.col("nr") == pl.col("round") + 1) & pl.col("nt").is_not_null() & pl.col("winner").is_not_null())
a = pl.when(pl.col("winner") == 2).then(1).otherwise(0)
b = pl.when(pl.col("winner") == 3).then(1).otherwise(0)
N = N.with_columns(ns=(pl.col("nt") == pl.col("t_fe") + a) & (pl.col("nct") == pl.col("ct_fe") + b),
                   sw=(pl.col("nt") == pl.col("ct_fe") + b) & (pl.col("nct") == pl.col("t_fe") + a))
N = N.with_columns(cls_b=pl.when(pl.col("ns") & pl.col("sw")).then(pl.lit("both"))
                   .when(pl.col("ns")).then(pl.lit("NS")).when(pl.col("sw")).then(pl.lit("SW")).otherwise(pl.lit("neither")))
NA = N.filter(pl.col("regA") & pl.col("day").is_in(LATE)).filter(pl.col("round").is_in([8, 12]))
NA = NA.join(CL.select(["match_key", "cls"]), on="match_key", how="inner")
print("\nC2. regime-A late: boundary class at 8 and 12 (recorded winner), by inversion class:")
print(NA.group_by(["cls", "round", "cls_b"]).len().sort(["cls", "round", "cls_b"]))

# C3. header test with per-class side convention
def side_x_class(cls_expr):
    # class X: halftime at 12 (T for r<=12, CT for 13..24, OT blocks of 6 with T for first 3)
    # class Y: halftime at 8 (T for r<=8, CT for 9..16, T for 17..24) -- ASSUMPTION for C3 only
    x12 = (pl.col("round") <= 12) | ((pl.col("round") >= 25) & (((pl.col("round") - 25) % 6) < 3))
    y8 = (pl.col("round") <= 8) | ((pl.col("round") >= 17) & (pl.col("round") <= 24))
    return pl.when(pl.col("cls") == "Y:r9").then(pl.when(y8).then(pl.lit(2)).otherwise(pl.lit(3))).otherwise(
        pl.when(x12).then(pl.lit(2)).otherwise(pl.lit(3)))


C = (R.filter(pl.col("regA")).group_by("match_key").agg(pl.len().alias("n_rounds"),
                                                       pl.col("t_starters_score_final").first().alias("hdr_t"),
                                                       pl.col("ct_starters_score_final").first().alias("hdr_ct")))
C = C.filter(pl.col("hdr_t") + pl.col("hdr_ct") == pl.col("n_rounds")).join(CL.select(["match_key", "cls"]), on="match_key", how="inner")
PC = R.filter(pl.col("regA")).join(C.select(["match_key", "hdr_t", "cls"]), on="match_key", how="inner")
PC = PC.with_columns(sideX=side_x_class(pl.col("cls")))
print(f"\nC3. complete regime-A matches in late or early days with class: {C.height}")
for name, flip in {"no flip": [], "flip first post-swap (13 for X, 9 for Y, plus 25/28 for X)": None}.items():
    if flip is None:
        w = pl.when((pl.col("cls") == "X:r13") & pl.col("round").is_in([13, 25, 28])).then(5 - pl.col("winner")) \
            .when((pl.col("cls") == "Y:r9") & (pl.col("round") == 9)).then(5 - pl.col("winner")) \
            .otherwise(pl.col("winner"))
    else:
        w = pl.col("winner")
    Q = PC.with_columns(wf=w).with_columns(xw=((pl.col("sideX") == pl.col("wf")).cast(pl.Int64)))
    tot = Q.group_by("match_key").agg(pl.col("xw").sum().alias("X"), pl.col("hdr_t").first().alias("hdr_t"),
                                     pl.col("cls").first())
    print(f"   {name}:")
    print(tot.with_columns(ok=(pl.col("X") == pl.col("hdr_t"))).group_by("cls").agg(pl.len().alias("n"), pl.col("ok").sum().alias("exact"),
                                                                                  (100.0 * pl.col("ok").mean()).round(2).alias("pct")).sort("cls"))
Q0 = PC.with_columns(xw=((pl.col("sideX") == pl.col("winner")).cast(pl.Int64)))
tot0 = Q0.group_by("match_key").agg(pl.col("xw").sum().alias("X"), pl.col("hdr_t").first().alias("hdr_t"), pl.col("cls").first())
print("   no flip by class:")
print(tot0.with_columns(ok=(pl.col("X") == pl.col("hdr_t"))).group_by("cls").agg(pl.len().alias("n"), pl.col("ok").sum().alias("exact")).sort("cls"))
