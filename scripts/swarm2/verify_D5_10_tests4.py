"""Verifier tests 4 for D5-10 (read only; caches in scratchpad only).

W1  Round count of the 2024 8->9 swap group (regime A and B): does it look like 16-round (MR8) matches?
W2  Overlap of the 8->9 swap group with the boundary-12 'no swap' class (regime A).
W3  Inspect the 51 early-diff matches: freeze-end chain and winners for rounds 1..14 and the header total, next to
    three ordinary regime-A matches without an 8->9 swap.

ASSUMPTION A11: a 16-round match (max_rounds 16) has its halftime after round 8, as in the 2026 bake-off tome.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_tests4.py
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
T, CT = 2, 3

PAIR = pl.read_parquet(SCR / "pairs.parquet").with_columns(tie0=(pl.col("t0") == pl.col("c0")))
FE = pl.read_parquet(SCR / "fe.parquet")
RE = pl.read_parquet(SCR / "re.parquet")
MR = pl.read_parquet(SCR / "mr.parquet")
hdr = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final"]).collect().rename({"key": "match_key"})
MR = MR.join(hdr.select(["match_key"]), on="match_key", how="left")
NM = RE.group_by("match_key").agg(pl.col("round").max().alias("N"), pl.len().alias("cnt"))
MR = MR.join(NM, on="match_key", how="left")
sw8 = PAIR.filter((pl.col("round") == 8) & (pl.col("cls") == "rec_H1")).select("match_key").unique().with_columns(swap8=pl.lit(True))
MR = MR.join(sw8, on="match_key", how="left").with_columns(swap8=pl.col("swap8").fill_null(False))

print("== W1 round count of the 8->9 swap group ==")
print(MR.filter(pl.col("swap8")).group_by(["regime", "N"]).len().sort(["regime", "N"]).head(30).rows())
print("regime A no-swap8 max label distribution (top 12):",
      MR.filter((pl.col("regime") == "A") & ~pl.col("swap8")).group_by("N").len().sort("len", descending=True).head(12).rows())
print("regime B (no swap8 group) max label distribution (top 12):",
      MR.filter((pl.col("regime") == "B") & ~pl.col("swap8")).group_by("N").len().sort("len", descending=True).head(12).rows())

print("\n== W2 boundary-12 class among the 8->9 swap group (regime A) ==")
p12 = PAIR.filter((pl.col("round") == 12) & (pl.col("regime") == "A")).join(
    MR.select(["match_key", "swap8"]), on="match_key", how="left")
print(p12.group_by(["swap8", "tie0", "cls"]).len().sort(["swap8", "tie0", "cls"]).rows())

print("\n== W3 early-diff matches (first diff at 9 or 10) inspected ==")
early_ids = ["2"]  # placeholder replaced below
Q = (RE.join(FE, on=["match_key", "round"], how="left").join(MR.select(["match_key", "regime", "swap8"]), on="match_key", how="inner")
     .filter((pl.col("regime") == "A") & ~pl.col("swap8")))
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
early = ck.filter(pl.col("diff") != 0).group_by("match_key").agg(pl.col("round").min().alias("first_r")).filter(pl.col("first_r") <= 12)
early_ids = early["match_key"].to_list()
ordinary = (ck.filter(pl.col("diff") == 0).group_by("match_key").agg(pl.len())
            .join(MR.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "N"]), on="match_key")
            .filter(pl.col("N") == 16).sort("match_key").head(0))
ordinary_ids = (ck.filter(pl.col("diff") == 0).select("match_key").unique().sort("match_key").head(3))["match_key"].to_list()


def show(mk, title):
    sub = (Q.filter(pl.col("match_key") == mk).select(["round", "t0", "c0", "winner_team_code", "diff"]).head(14))
    h = MR.filter(pl.col("match_key") == mk).select(["N", "t_starters_score_final", "ct_starters_score_final"]).row(0)
    p8 = PAIR.filter((pl.col("match_key") == mk) & pl.col("round").is_in([8, 9, 12, 13])).select(
        ["round", "t0", "c0", "t1", "c1", "winner_team_code", "cls", "tie0"]).sort("round")
    print(f"\n-- {title}: {mk.split('/')[-2][:8]}  N={h[0]}  header T-start={h[1]}  CT-start={h[2]}")
    print(sub.rows())
    print("   boundary 8, 9, 12, 13:", p8.rows())


for mk in early_ids[:3]:
    show(mk, "early-diff")
for mk in ordinary_ids[:2]:
    show(mk, "ordinary (no diff)")
print(f"\nearly-diff matches listed: {len(early_ids)}; their boundary-8 classes:")
print(PAIR.filter((pl.col("round") == 8) & pl.col("match_key").is_in(early_ids)).group_by(["tie0", "cls"]).len().rows())
print("early-diff matches: header total vs N (count):",
      MR.filter(pl.col("match_key").is_in(early_ids)).with_columns(
          (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") - pl.col("N")).alias("d")).group_by("d").len().sort("d").rows())
