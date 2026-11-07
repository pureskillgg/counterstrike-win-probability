"""Verifier for D5-8, round-9 check: is the regime-A round-9 step disagreement real or a field artifact?

  N1. regime A late: round-9 bomb rows vs step disagreement (cross-tab per round).
  N2. header test on complete regime-A matches with flip sets that include or exclude round 9.
  N3. for the round-9 step-disagreement matches: their round-8 and round-12 step/bomb status (is it a halftime-8 pattern?).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_8_nine.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_8")
R = pl.read_parquet(SCR / "rounds2024.parquet").rename({"winner_team_code": "winner"})
H = pl.read_parquet(SCR / "header2024.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
RA = R.filter(pl.col("regA") & pl.col("day").is_in(LATE))

R9 = RA.filter(pl.col("round") == 9).with_columns(
    step_dis=pl.when(pl.col("step_side").is_null()).then(pl.lit("none"))
    .when(pl.col("step_side") != pl.col("winner")).then(pl.lit("DIS")).otherwise(pl.lit("agree")),
    bomb_s=pl.when(pl.col("bomb_side").is_null()).then(pl.lit("nobomb"))
    .when(pl.col("bomb_side") != pl.col("winner")).then(pl.lit("DIS")).otherwise(pl.lit("agree")))
print("N1. regime A late, round 9: step status x bomb status (matches):")
print(R9.group_by(["step_dis", "bomb_s"]).len().sort(["step_dis", "bomb_s"]))

# per-match frame for round 8, 9, 12, 13 (recorded winner and step/bomb status)
def status(r):
    s = RA.filter(pl.col("round") == r).select(["match_key", pl.col("winner").alias(f"w{r}"),
                                              pl.col("step_side").alias(f"s{r}"), pl.col("bomb_side").alias(f"b{r}")])
    return s
M = status(8).join(status(9), on="match_key", how="full", coalesce=True).join(status(12), on="match_key", how="full", coalesce=True)\
    .join(status(13), on="match_key", how="full", coalesce=True)
M = M.with_columns(d9=(pl.col("s9").is_not_null() & (pl.col("s9") != pl.col("w9"))))
print("\nN3. round-9 step-disagreement matches (regime A late): counts by round-8 status and round-12/13 step status")
S9 = M.filter(pl.col("d9"))
print("   n matches with round-9 step disagreement:", S9.height)
print(S9.with_columns(
    st8=pl.when(pl.col("s8").is_null()).then(pl.lit("none")).when(pl.col("s8") == pl.col("w8")).then(pl.lit("agree")).otherwise(pl.lit("DIS")),
    st13=pl.when(pl.col("s13").is_null()).then(pl.lit("none")).when(pl.col("s13") == pl.col("w13")).then(pl.lit("agree")).otherwise(pl.lit("DIS")))
      .group_by(["st8", "st13"]).len().sort(["st8", "st13"]))
print("   bomb rows at round 9 inside these matches:", S9.filter(pl.col("b9").is_not_null()).height,
      " of which bomb != winner:", S9.filter(pl.col("b9").is_not_null() & (pl.col("b9") != pl.col("w9"))).height)

# ---- N2. header test on complete regime-A matches ----
def side_x(r):
    return pl.when((pl.col("round") <= 12) | ((pl.col("round") >= 25) & (((pl.col("round") - 25) % 6) < 3))).then(pl.lit(2)).otherwise(pl.lit(3))


P = R.filter(pl.col("regA")).with_columns(sideX=side_x("round"))
cnt = P.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("t_starters_score_final").first().alias("hdr_t"),
                                  pl.col("ct_starters_score_final").first().alias("hdr_ct"))
C = cnt.filter(pl.col("hdr_t") + pl.col("hdr_ct") == pl.col("n_rounds"))
PC = P.join(C.select(["match_key", "hdr_t", "n_rounds"]), on="match_key")
print(f"\nN2. complete regime-A matches: {C.height}")
FLIPS = {"none": [], "13": [13], "13,25,28": [13, 25, 28], "9,13,25,28": [9, 13, 25, 28], "9,13": [9, 13]}
for name, rounds in FLIPS.items():
    w = pl.when(pl.col("round").is_in(rounds)).then(5 - pl.col("winner")).otherwise(pl.col("winner")) if rounds else pl.col("winner")
    Q = PC.with_columns(xw=((pl.col("sideX") == w).cast(pl.Int64)))
    tot = Q.group_by("match_key").agg(pl.col("xw").sum().alias("X"), pl.col("hdr_t").first().alias("hdr_t"))
    ok = (tot["X"] == tot["hdr_t"]).sum()
    print(f"   flip {name:12s}: exact {ok:5d} / {tot.height} ({100.0 * ok / tot.height:.2f}%)")

# residual mismatches under flip {13,25,28}: which matches fail, and do they have round-9 step disagreement?
w = pl.when(pl.col("round").is_in([13, 25, 28])).then(5 - pl.col("winner")).otherwise(pl.col("winner"))
Q = PC.with_columns(xw=((pl.col("sideX") == w).cast(pl.Int64)), wf=w)
tot = Q.group_by("match_key").agg(pl.col("xw").sum().alias("X"), pl.col("hdr_t").first().alias("hdr_t"),
                                 pl.col("n_rounds").first())
bad = tot.filter(pl.col("X") != pl.col("hdr_t")).select("match_key")
print(f"   residual under flip 13,25,28: {bad.height} matches")
d9 = M.filter(pl.col("d9")).select("match_key")
print("   residual matches with round-9 step disagreement:", bad.join(d9, on="match_key", how="inner").height)
print("   residual match round counts:", tot.filter(pl.col("X") != pl.col("hdr_t")).group_by("n_rounds").len().sort("n_rounds").to_dicts())
