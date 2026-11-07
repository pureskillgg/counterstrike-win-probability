"""Verifier tests 2 for D5-10. Reads the cached tables from verify_D5_10_main.py (scratchpad) and the 2024
header (read only). Nothing is written to the project.

U1  Winner-free check: team A's final total from freeze-end fields alone. Under the standard side model (team A on T
    for rounds 1..12, on CT for 13..N in regulation) the freeze-end score of team A at label N plus at most one point
    for round N must equal the header total. d = header_A - fe_A(N) should be 0 or 1. This uses no round_end winner
    except none at all (the last round's winner is allowed as the single 0/1).
U2  Convention brute force for regulation matches (N <= 24): which swap points and which inverted rounds reproduce the
    header total. Groups: regime A without 8->9 swap (claim population), regime A with 8->9 swap (excluded by claim).
U3  Map and day breakdown of the round-13 signal (non-tie pre-state, regime A and B), to test whether the signal is
    map-specific.
U4  First differing label of the 52 matches that differ before label 14, and whether they carry the round-9 signal.
U5  OT matches (N > 24): header total under three swap schedules, with and without inverted rounds 13 or 25.

ASSUMPTION A8: header t_starters_score_final is team A's final score (team that started on T), as used by bakeoff.py.
ASSUMPTION A9: regulation halftime at round 12 unless a test says otherwise.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_tests2.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3

PAIR = pl.read_parquet(SCR / "pairs.parquet").with_columns(tie0=(pl.col("t0") == pl.col("c0")))
FE = pl.read_parquet(SCR / "fe.parquet")
RE = pl.read_parquet(SCR / "re.parquet")
MR = pl.read_parquet(SCR / "mr.parquet")
HDR = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(str(BASE / "header" / "dataframe_*")))]) \
    if False else pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]).collect().rename({"key": "match_key"})
MR = MR.join(HDR.select(["match_key", "map_name"]), on="match_key", how="left")
RE_cnt = RE.group_by("match_key").agg(pl.len().alias("cnt"), pl.col("round").min().alias("minr"), pl.col("round").max().alias("maxr"))
DIAG = MR.join(RE_cnt, on="match_key", how="left").with_columns(dsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
print("== diagnostics: header total vs round labels ==")
print(DIAG.with_columns((pl.col("dsum") - pl.col("maxr")).clip(-3, 3).alias("dsum_minus_maxr"),
                        (pl.col("dsum") - pl.col("cnt")).clip(-3, 3).alias("dsum_minus_cnt"))
      .group_by(["regime", "dsum_minus_maxr"]).len().sort(["regime", "dsum_minus_maxr"]))
print("min label by regime:", DIAG.group_by(["regime", "minr"]).len().sort(["regime", "minr"]).head(8).rows())
print("count of round_end rows minus max label, by regime:")
print(DIAG.with_columns((pl.col("cnt") - pl.col("maxr")).alias("cnt_minus_max")).group_by(["regime", "cnt_minus_max"]).len().sort(["regime", "cnt_minus_max"]).head(12))
print("map_name values (top 12):", MR["map_name"].value_counts().sort("count", descending=True).head(12).rows())

# swap8 group (claim excludes these)
swap8 = PAIR.filter((pl.col("round") == 8) & (pl.col("cls") == "rec_H1")).select("match_key").unique().with_columns(
    swap8=pl.lit(True))
MR = MR.join(swap8, on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))
MR = MR.with_columns(grp=pl.when(pl.col("regime") == "B").then(pl.lit("B"))
                     .when(pl.col("swap8")).then(pl.lit("A_swap8"))
                     .otherwise(pl.lit("A_noswap8")))
print("groups:", MR["grp"].value_counts().sort("grp").rows())

# per match: N and freeze-end at N
NM = RE.group_by("match_key").agg(pl.col("round").max().alias("N"), pl.len().alias("n_rows"))
FEN = FE.join(NM, on="match_key", how="inner").filter(pl.col("round") == pl.col("N")).select(
    ["match_key", "N", "t0", "c0"])
U = FEN.join(MR.select(["match_key", "regime", "grp", "map_name", "t_starters_score_final", "ct_starters_score_final",
                        "match_date"]), on="match_key", how="left")
U = U.with_columns(hsum_minus_N=(pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") - pl.col("N")))

# ---------------------------------------------------------------- U1 winner-free header check (regulation, N <= 24)
print("\n== U1 winner-free check: header_A - fe_A(N), regulation N <= 24 ==")
Ur = U.filter(pl.col("N") <= 24).with_columns(
    fe_A=pl.when(pl.col("N") <= 12).then(pl.col("t0")).otherwise(pl.col("c0")))
Ur = Ur.with_columns(dA=pl.col("t_starters_score_final") - pl.col("fe_A"))
print(Ur.group_by(["grp", "dA"]).len().sort(["grp", "dA"]).filter(pl.col("dA").is_between(-3, 4)))
print("regulation matches (N <= 24) by group:", Ur.group_by("grp").len().sort("grp").rows())
print("share with header_A - fe_A(N) in {0,1}, by group:")
print(Ur.group_by("grp").agg(pl.len().alias("n"), pl.col("dA").is_in([0, 1]).sum().alias("in_0_1"),
                             (pl.col("dA").is_in([0, 1]).mean()).round(4).alias("share")).sort("grp"))

# ---------------------------------------------------------------- U2 convention brute force (regulation)
print("\n== U2 convention brute force (regulation N <= 24): team A total vs header ==")
R = (RE.join(MR.select(["match_key", "grp", "t_starters_score_final"]), on="match_key", how="inner")
     .filter(pl.col("grp").is_in(["A_noswap8", "A_swap8"])))
R = R.join(NM.filter(pl.col("N") <= 24).select("match_key"), on="match_key", how="inner")
# contiguous labels 1..N only
ok_mk = NM.filter(pl.col("N") <= 24).join(RE.group_by("match_key").agg(pl.len().alias("cnt")), on="match_key")
ok_mk = ok_mk.filter(pl.col("cnt") == pl.col("N")).select("match_key")
R = R.join(ok_mk, on="match_key", how="inner").select(["match_key", "grp", "round", "winner_team_code", "t_starters_score_final"])
print("matches in U2 (contiguous 1..N, N <= 24):", R["match_key"].n_unique(), R.group_by("grp").agg(pl.col("match_key").n_unique()).sort("grp").rows())

combos = {
    "swap{12}": [12], "swap{8}": [8], "swap{8,12}": [8, 12], "swap{}": [],
    "swap{12,24}": [12, 24],
}
invs = {"inv{}": [], "inv{9}": [9], "inv{13}": [13], "inv{9,13}": [9, 13]}


def team_a_total(df, swaps, inv):
    nsw = pl.lit(0)
    for s in swaps:
        nsw = nsw + (pl.col("round") > s).cast(pl.Int64)
    onT = (nsw % 2 == 0)
    w = pl.col("winner_team_code")
    if inv:
        w = pl.when(pl.col("round").is_in(inv) & (w == T)).then(pl.lit(CT)).when(
            pl.col("round").is_in(inv) & (w == CT)).then(pl.lit(T)).otherwise(w)
    aw = pl.when(onT & (w == T)).then(1).when((~onT) & (w == CT)).then(1).otherwise(0)
    return df.with_columns(aw=aw).group_by("match_key").agg(pl.col("aw").sum().alias("A"),
                                                              pl.col("t_starters_score_final").first().alias("hT"),
                                                              pl.col("grp").first())


rows = []
for cn, cs in combos.items():
    for iname, ins in invs.items():
        tot = team_a_total(R, cs, ins)
        g = tot.group_by("grp").agg((pl.col("A") == pl.col("hT")).sum().alias("match"), pl.len().alias("n"))
        for r in g.iter_rows(named=True):
            rows.append((r["grp"], cn, iname, r["match"], r["n"]))
bf = pl.DataFrame(rows, schema=["grp", "swaps", "inv", "header_match", "n"], orient="row").sort(["grp", "header_match"], descending=[False, True])
print(bf.group_by("grp").head(6))

# ---------------------------------------------------------------- U3 map / day breakdown of the round-13 signal
print("\n== U3 round-13 signal by map (regime A, non-tie pre-state, boundary 13) ==")
b13 = PAIR.filter((pl.col("round") == 13) & (~pl.col("tie0"))).join(MR.select(["match_key", "map_name", "grp", "match_date"]),
                                                                    on="match_key", how="left")
b13 = b13.with_columns(sig=pl.when(pl.col("H0f") & ~pl.col("H0")).then(pl.lit("flip"))
                       .when(pl.col("H0") & ~pl.col("H0f")).then(pl.lit("rec"))
                       .otherwise(pl.lit("other")))
print("boundary 13 non-tie signal by regime:")
print(b13.group_by(["regime", "sig"]).len().sort(["regime", "sig"]))
print("regime A, non-tie, flip share by map (maps with >= 20 pairs):")
mp = b13.filter(pl.col("regime") == "A").group_by("map_name").agg(
    pl.len().alias("n"), (pl.col("sig") == "flip").sum().alias("flip"), (pl.col("sig") == "rec").sum().alias("rec")).filter(
    pl.col("n") >= 20).with_columns((pl.col("flip") / (pl.col("flip") + pl.col("rec"))).round(3).alias("flip_share")).sort("n", descending=True)
print(mp)
print("regime A, non-tie: flip vs rec by day (first 25 days):")
print(b13.filter(pl.col("regime") == "A").group_by("match_date").agg(
    pl.len().alias("n"), (pl.col("sig") == "flip").sum().alias("flip"), (pl.col("sig") == "rec").sum().alias("rec")).sort("match_date").head(25))

# ---------------------------------------------------------------- U4 early first-diff matches
print("\n== U4 matches whose first differing label is 9 or 10 (claim population) ==")
Q = (RE.join(FE, on=["match_key", "round"], how="left")
     .join(MR.select(["match_key", "regime", "grp"]), on="match_key", how="inner").filter(pl.col("grp") == "A_noswap8"))
FE2 = FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t0").alias("t2"), pl.col("c0").alias("c2")])
Q = Q.join(FE2, on="match_key", how="left").sort(["match_key", "round"])
onT = (pl.col("round") <= 12) | pl.col("round").is_between(25, 27)
Q = Q.with_columns(a_fe=pl.when(onT).then(pl.col("t0")).otherwise(pl.col("c0")),
                   a_won=pl.when(pl.col("round") == 1).then(pl.when(onT).then(pl.col("t2")).otherwise(pl.col("c2")))
                   .otherwise(pl.when(onT & (pl.col("winner_team_code") == T)).then(1)
                              .when((~onT) & (pl.col("winner_team_code") == CT)).then(1).otherwise(0)))
Q = Q.with_columns(cum_before=pl.col("a_won").fill_null(0).cum_sum().over("match_key") - pl.col("a_won").fill_null(0))
Q = Q.with_columns(diff=pl.col("a_fe") - pl.col("cum_before"))
ck = Q.filter(pl.col("a_fe").is_not_null() & (pl.col("round") >= 2) & pl.col("winner_team_code").is_not_null())
first = ck.filter(pl.col("diff") != 0).group_by("match_key").agg(pl.col("round").min().alias("first_r"))
early = first.filter(pl.col("first_r") <= 12)
print("matches with first diff <= 12 (claim population):", early.height, "; by first label:", early.group_by("first_r").len().sort("first_r").rows())
flip9 = PAIR.filter((pl.col("round") == 9) & (~pl.col("tie0")) & pl.col("H0f") & ~pl.col("H0")).select("match_key").unique()
print("of these, with a non-tie flip signature at boundary 9:", early.join(flip9, on="match_key", how="inner").height)
flip13 = PAIR.filter((pl.col("round") == 13) & (~pl.col("tie0")) & pl.col("H0f") & ~pl.col("H0")).select("match_key").unique()
print("of these, with a non-tie flip signature at boundary 13:", early.join(flip13, on="match_key", how="inner").height)
print("regime A noswap8 matches with a non-tie flip signature at boundary 9 (any):",
      flip9.join(MR.filter(pl.col("grp") == "A_noswap8").select("match_key"), on="match_key", how="inner").height)
print("regime A swap8 matches with a non-tie flip signature at boundary 9:",
      flip9.join(MR.filter(pl.col("grp") == "A_swap8").select("match_key"), on="match_key", how="inner").height)

# ---------------------------------------------------------------- U5 OT
print("\n== U5 OT matches (N > 24): header total under swap schedules ==")
OT = NM.filter(pl.col("N") > 24).select("match_key")
RO = (RE.join(OT, on="match_key", how="inner").join(MR.select(["match_key", "grp", "t_starters_score_final"]),
                                                   on="match_key", how="left"))
RO = RO.join(NM.select(["match_key", "N"]), on="match_key", how="left")
ot_combos = {"D5 swap{12,24,27}": [12, 24, 27], "bakeoff swap{12,27}": [12, 27],
             "swap{12,24,27,30}": [12, 24, 27, 30], "swap{12,27,33}": [12, 27, 33]}
for cn, cs in ot_combos.items():
    for iname, ins in {"inv{}": [], "inv{13}": [13], "inv{25}": [25], "inv{13,25}": [13, 25]}.items():
        tot = team_a_total(RO.select(["match_key", "round", "winner_team_code", "grp", "t_starters_score_final"]), cs, ins)
        g = tot.group_by("grp").agg((pl.col("A") == pl.col("hT")).sum().alias("match"), pl.len().alias("n"))
        print(f"  {cn:<22} {iname:<10}", {r["grp"]: f"{r['match']}/{r['n']}" for r in g.iter_rows(named=True)})
