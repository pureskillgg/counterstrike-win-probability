"""Verifier tests for D5-10, run on the tables cached by verify_D5_10_main.py (scratchpad, read only).

T1  Tie-aware boundary test. If the pre-round score is tied (t0 == c0), the hypotheses
    "H1 with recorded winner" and "H0 with inverted winner" predict identical fields, so tied states
    cannot separate a side swap from a winner inversion. Counts are split by tie status.
T2  Header test (regulation matches only, N <= 24, complete): team A final = header t_starters_score_final.
    Compare with (a) the D5 team-A cumulative from recorded winners, (b) the same with round 13 inverted,
    and (c) the same with the MR8 halftime convention (swap after round 8), for matches that show an 8->9 swap.
T3  D5_10 replication: D5_17 logic (regime A, no 8->9 swap) rebuilt from the tome tables; counts vs the claim
    (2,223 matches, 41,818 states, 15,614 differing states, 2,103 matches). First differing label distribution.
    Then the same check after inverting round 13 for regime A matches flagged by the non-tie boundary test.
T4  Team totals from the header versus the round-end winners for OT matches (where the D5 convention is
    not valid beyond round 27), reported only as counts.

ASSUMPTION A6: team A = starters on T in round 1; header t_starters_score_final = team A final score.
ASSUMPTION A7: regulation halftime after round 12 (MR12); the MR8 alternative is tested separately, not assumed.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_tests.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
T, CT = 2, 3

PAIR = pl.read_parquet(SCR / "pairs.parquet")
FE = pl.read_parquet(SCR / "fe.parquet")
RE = pl.read_parquet(SCR / "re.parquet")
MR = pl.read_parquet(SCR / "mr.parquet")
print(f"loaded: pairs {PAIR.height}; fe {FE.height}; re {RE.height}; matches {MR.height}")

# ---------------------------------------------------------------- T1 tie-aware boundary test
print("\n== T1 tie-aware boundary test ==")
PAIR = PAIR.with_columns(
    tie0=(pl.col("t0") == pl.col("c0")),
    tie1=(pl.col("t1") == pl.col("c1")),
    fit=(pl.when(pl.col("H0")).then(pl.lit("rec_H0 ")).otherwise(pl.lit(""))
         + pl.when(pl.col("H1")).then(pl.lit("rec_H1 ")).otherwise(pl.lit(""))
         + pl.when(pl.col("H0f")).then(pl.lit("flip_H0 ")).otherwise(pl.lit(""))
         + pl.when(pl.col("H1f")).then(pl.lit("flip_H1 ")).otherwise(pl.lit(""))))
PAIR = PAIR.filter(pl.col("winner_team_code").is_in([T, CT]))
print("pairs with a T/CT winner:", PAIR.height)
print("tie-status of pre-round score, regime A, labels 1..30 (tied / not tied):")
print(PAIR.filter(pl.col("regime") == "A").group_by(["round", "tie0"]).len()
      .pivot(on="tie0", index="round", values="len").fill_null(0).sort("round").filter(pl.col("round") <= 30))

for lab in [8, 9, 12, 13, 14, 24, 25, 27]:
    sub = PAIR.filter((pl.col("regime") == "A") & (pl.col("round") == lab))
    print(f"\nregime A boundary {lab}->{lab + 1}: fit patterns, NON-TIE pre-state only")
    print(sub.filter(~pl.col("tie0")).group_by("fit").len().sort("len", descending=True).head(10))
    print(f"  tied pre-state: {sub.filter(pl.col('tie0')).height} pairs (fit patterns not identifiable)")
sub = PAIR.filter((pl.col("regime") == "B") & (pl.col("round") == 13))
print("\nregime B boundary 13->14, NON-TIE pre-state:")
print(sub.filter(~pl.col("tie0")).group_by("fit").len().sort("len", descending=True).head(10))

# per-match round-13 inversion evidence in regime A (non-tie pre-state, exact fit)
inv13 = (PAIR.filter((pl.col("round") == 13) & (~pl.col("tie0")) & (pl.col("regime") == "A"))
         .with_columns(
             inv13_flip_H0=pl.col("H0f") & ~pl.col("H0") & ~pl.col("H1"),
             inv13_flip_H1=pl.col("H1f") & ~pl.col("H0") & ~pl.col("H1") & ~pl.col("H0f"),
             rec_ok=(pl.col("H0") | pl.col("H1")) & ~(pl.col("H0f") | pl.col("H1f")))
         .select(["match_key", "inv13_flip_H0", "inv13_flip_H1", "rec_ok"]))
print("\nregime A round-13 boundary, non-tie pre-state, exact-fit classes:",
      inv13.select([pl.col("inv13_flip_H0").sum().alias("flip_H0_only"),
                    pl.col("inv13_flip_H1").sum().alias("flip_H1_only"),
                    pl.col("rec_ok").sum().alias("recorded_fits_only"), pl.len().alias("pairs")]).row(0))

# ---------------------------------------------------------------- build per-round table
R = (RE.join(FE, on=["match_key", "round"], how="left")
     .join(MR.select(["match_key", "regime", "t_starters_score_final", "ct_starters_score_final"]),
           on="match_key", how="left")
     .sort(["match_key", "round"]))
mstat = R.group_by("match_key").agg(pl.col("round").max().alias("N"), pl.len().alias("n_rows"),
                                    pl.col("t0").is_not_null().sum().alias("n_fe"),
                                    (pl.col("round").max() == pl.len()).alias("contig"),
                                    pl.col("regime").first(), pl.col("t_starters_score_final").first().alias("hT"),
                                    pl.col("ct_starters_score_final").first().alias("hC"))
mstat = mstat.with_columns(hsum_eq_N=(pl.col("hT") + pl.col("hC") == pl.col("N")))
print("\nmatch completeness: matches with winner rows; contiguous labels by regime:")
print(mstat.group_by("regime").agg(pl.len().alias("matches"), pl.col("contig").sum().alias("contig_labels"),
                                   (pl.col("N") <= 24).sum().alias("N_le_24"),
                                   (pl.col("N") > 24).sum().alias("N_gt_24"),
                                   pl.col("hsum_eq_N").sum().alias("header_sum_eq_rounds")).sort("regime"))

# ---------------------------------------------------------------- T2 header test, regulation
print("\n== T2 header test (regulation, N <= 24, contiguous labels, header sum == rounds) ==")


def cum_team_a(df, sw_round, invert13=False):
    """Team A (starters on T in round 1) total: on T for rounds <= sw_round, CT after. Recorded winners,
    optionally with round 13 inverted."""
    w = pl.col("winner_team_code")
    if invert13:
        w = pl.when(pl.col("round") == 13).then(
            pl.when(w == T).then(pl.lit(CT)).when(w == CT).then(pl.lit(T)).otherwise(w)).otherwise(w)
    onT = pl.col("round") <= sw_round
    return df.with_columns(aw=pl.when(onT & (w == T)).then(1).when((~onT) & (w == CT)).then(1).otherwise(0)) \
        .group_by("match_key").agg(pl.col("aw").sum().alias("A"))


reg = mstat.filter((pl.col("N") <= 24) & pl.col("contig") & pl.col("hsum_eq_N")).select(["match_key", "regime", "hT", "N"])
Rr = R.join(reg.select("match_key"), on="match_key", how="inner").select(["match_key", "round", "winner_team_code"])
A12 = cum_team_a(Rr, 12).join(reg, on="match_key")
A12i = cum_team_a(Rr, 12, invert13=True).rename({"A": "Ai"})
A8 = cum_team_a(Rr, 8).rename({"A": "A8"})
Q = A12.join(A12i, on="match_key").join(A8, on="match_key")
Q = Q.with_columns(ok12=pl.col("A") == pl.col("hT"), ok12i=pl.col("Ai") == pl.col("hT"), ok8=pl.col("A8") == pl.col("hT"))
print("regulation matches with contiguous labels and header sum == rounds, by regime:")
print(Q.group_by("regime").agg(pl.len().alias("matches"), pl.col("ok12").sum().alias("MR12_recorded_matches_header"),
                               pl.col("ok12i").sum().alias("MR12_round13_inverted_matches_header"),
                               (pl.col("ok12") | pl.col("ok12i")).sum().alias("either"),
                               pl.col("ok8").sum().alias("MR8_convention_matches_header")).sort("regime"))
# matches that agree with header only after inverting round 13
Q2 = Q.join(PAIR.filter((pl.col("round") == 8) & (pl.col("cls") == "rec_H1")).select("match_key").with_columns(swap8=pl.lit(True)),
            on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))
print("\nregulation, regime A, by 8->9 swap flag (rec_H1 at boundary 8):")
print(Q2.filter(pl.col("regime") == "A").group_by("swap8").agg(
    pl.len().alias("matches"), pl.col("ok12").sum().alias("ok_recorded"), pl.col("ok12i").sum().alias("ok_inv13"),
    (~pl.col("ok12") & ~pl.col("ok12i")).sum().alias("neither"), pl.col("ok8").sum().alias("ok_MR8")).sort("swap8"))
print("regulation, regime B, by 8->9 swap flag:")
print(Q2.filter(pl.col("regime") == "B").group_by("swap8").agg(
    pl.len().alias("matches"), pl.col("ok12").sum().alias("ok_recorded"), pl.col("ok12i").sum().alias("ok_inv13"),
    (~pl.col("ok12") & ~pl.col("ok12i")).sum().alias("neither"), pl.col("ok8").sum().alias("ok_MR8")).sort("swap8"))

# ---------------------------------------------------------------- T3 D5_10 / D5_17 replication
print("\n== T3 D5_17 replication (regime A, no 8->9 swap) ==")
swap8_set = PAIR.filter((pl.col("round") == 8) & (pl.col("cls") == "rec_H1")).select("match_key").unique()
MRA = MR.join(swap8_set.with_columns(has_swap8=pl.lit(True)), on="match_key", how="left").with_columns(
    has_swap8=pl.col("has_swap8").fill_null(False))
print("regime A matches: total", MRA.filter(pl.col("regime") == "A").height,
      "; regime A with 8->9 swap:", MRA.filter((pl.col("regime") == "A") & pl.col("has_swap8")).height,
      "; regime B with 8->9 swap:", MRA.filter((pl.col("regime") == "B") & pl.col("has_swap8")).height)

FE2 = FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("t2"), pl.col("c0").alias("c2")])


def d5_check(invert_set=None, label=""):
    base = (RE.join(FE, on=["match_key", "round"], how="left")
            .join(MRA.select(["match_key", "regime", "has_swap8"]), on="match_key", how="inner")
            .filter((pl.col("regime") == "A") & (~pl.col("has_swap8"))).join(FE2, on="match_key", how="left")
            .sort(["match_key", "round"]))
    onT = (pl.col("round") <= 12) | pl.col("round").is_between(25, 27)
    w = pl.col("winner_team_code")
    if invert_set is not None:
        w = pl.when(pl.col("match_key").is_in(invert_set) & (pl.col("round") == 13)).then(
            pl.when(w == T).then(pl.lit(CT)).when(w == CT).then(pl.lit(T)).otherwise(w)).otherwise(w)
    base = base.with_columns(a_fe=pl.when(onT).then(pl.col("t0")).otherwise(pl.col("c0")),
                             a_won=pl.when(pl.col("round") == 1).then(pl.when(onT).then(pl.col("t2")).otherwise(pl.col("c2")))
                             .otherwise(pl.when(onT & (w == T)).then(1).when((~onT) & (w == CT)).then(1).otherwise(0)))
    base = base.with_columns(cum_before=pl.col("a_won").fill_null(0).cum_sum().over("match_key") - pl.col("a_won").fill_null(0))
    base = base.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
    ck = base.filter(pl.col("a_fe").is_not_null() & (pl.col("round") >= 2) & pl.col("winner_team_code").is_not_null())
    nm = ck["match_key"].n_unique()
    st = ck.height
    dstates = ck.filter(pl.col("diff") != 0)
    dm = dstates["match_key"].n_unique()
    print(f"{label}: matches {nm}; states {st}; diff!=0 states {dstates.height}; matches with any diff!=0 {dm}")
    return ck, dstates


ck, dstates = d5_check(label="D5 replication, recorded winners")
print("first differing label per match (distribution, first 20):")
first = dstates.group_by("match_key").agg(pl.col("round").min().alias("first_r"))
print(first.group_by("first_r").len().sort("first_r").head(20))
print("differing states by label (first 20):")
print(dstates.group_by("round").agg(pl.len().alias("states")).sort("round").head(20))

# T3b: invert round 13 for the matches with a non-tie flip_H0 or flip_H1 signature at 13
flip13 = inv13.filter(pl.col("inv13_flip_H0") | pl.col("inv13_flip_H1")).select("match_key")
print(f"\nregime A matches with a non-tie flip signature at round 13: {flip13.height}")
ck2, dstates2 = d5_check(invert_set=flip13["match_key"].to_list(), label="D5 replication, round 13 inverted where flip signature")
print("first differing label per match after correction:")
print(dstates2.group_by("match_key").agg(pl.col("round").min().alias("first_r")).group_by("first_r").len().sort("first_r").head(20))

# T3c: every regime A match with winner rows, round 13 inverted for all (universal correction)
allA = MRA.filter(pl.col("regime") == "A").select("match_key")
ck3, dstates3 = d5_check(invert_set=allA["match_key"].to_list(), label="D5 replication, round 13 inverted for ALL regime A (universal)")

# ---------------------------------------------------------------- T4 OT matches, header total with the D5 convention
print("\n== T4 OT matches (N > 24), header total check (team A on T for 1..12 and 25..27, CT after) ==")
ot = mstat.filter(pl.col("N") > 24).select(["match_key", "regime", "hT", "hC", "N"])
Ro = R.join(ot.select("match_key"), on="match_key", how="inner").select(["match_key", "round", "winner_team_code"])
onT_expr = (pl.col("round") <= 12) | pl.col("round").is_between(25, 27)
Ro = Ro.with_columns(aw=pl.when(onT_expr & (pl.col("winner_team_code") == T)).then(1)
                     .when((~onT_expr) & (pl.col("winner_team_code") == CT)).then(1).otherwise(0))
Ao = Ro.group_by("match_key").agg(pl.col("aw").sum().alias("A")).join(ot, on="match_key")
Ao = Ao.with_columns(ok=pl.col("A") == pl.col("hT"))
print(Ao.group_by("regime").agg(pl.len().alias("OT_matches"), pl.col("ok").sum().alias("A_equals_header_T")).sort("regime"))
print("OT matches by N (regime A):")
print(ot.filter(pl.col("regime") == "A").group_by("N").len().sort("N").head(12))

# ---------------------------------------------------------------- output for the log
print("\nrecorded winners vs freeze-end steps: summary for the claim numbers")
print("claim: regime A matches without 8->9 swap = 2,223; states 41,818; diff!=0 states 15,614; matches 2,103")
