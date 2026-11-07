r"""D5-8 verifier: independent check against header final scores (t_starters_score_final, ct_starters_score_final).

For matches with contiguous rounds 1..N and N == t_final + ct_final (no overtime, complete), the T-starters team's total
is W_T = #(label==2, r<=12) + #(label==3, 13<=r<=24) under the MR12 rule (ASSUMPTION: sides swap once after round 12).
Compare W_T with t_starters_score_final under (a) the raw label and (b) the label with round 13 flipped.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_finalscore.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")


def scan(ch):
    fd = os.path.join(BASE, f"basic_{ch}{SUF}")
    return pl.scan_parquet(sorted(glob.glob(os.path.join(fd, "dataframe_*"))), missing_columns="insert",
                           extra_columns="ignore")


# header lives in a single file under the 'header' folder
hfiles = sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))
hdr = pl.concat([pl.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final",
                                             "match_type", "match_date"]) for f in hfiles], how="vertical_relaxed")
hdr = hdr.unique("match_id")
hdr = hdr.with_columns(mid=pl.col("match_id"))
print("header rows:", hdr.height, " unique match_id:", hdr["match_id"].n_unique())
print("header match_type:", hdr.group_by("match_type").len().sort("len", descending=True).head(5).rows())

R = pl.read_parquet(OUT / "rounds_v2.parquet")
M = pl.read_parquet(OUT / "matches_v.parquet")
# completeness: contiguous rounds 1..N
agg = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                 pl.col("round").n_unique().alias("nr"), pl.col("winner").is_not_null().sum().alias("nw"))
agg = agg.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")))
agg = agg.with_columns(mid=pl.col("match_key").str.split("/").list.get(4))
X = agg.join(hdr.select(["mid", "t_starters_score_final", "ct_starters_score_final", "match_type"]), on="mid", how="left").join(M.select(["match_key", "regime", "day"]), on="match_key", how="left")
X = X.with_columns(final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
X = X.with_columns(sum_ok=pl.col("final_sum") == pl.col("rmax"))
print("\nmatches:", X.height, " header found:", int(X.filter(pl.col("t_starters_score_final").is_not_null()).height))
print("contiguous 1..N:", int(X.filter(pl.col("contig")).height))
print("header sum == rounds present (contig):", X.filter(pl.col("contig") & pl.col("sum_ok")).group_by("regime").len().rows())
print("contig but sum mismatch:", X.filter(pl.col("contig") & ~pl.col("sum_ok")).height)
print("header final sum distribution (contig):", X.filter(pl.col("contig")).group_by("final_sum").len().sort("final_sum").head(40).rows())

# team-identity wins under raw label, MR12 rule, rounds 1..24 only (no OT)
C = X.filter(pl.col("contig") & pl.col("sum_ok") & (pl.col("rmax") == 24)).select(["match_key", "regime", "day", "t_starters_score_final",
                                                       "ct_starters_score_final"])
RR = R.join(C.select("match_key"), on="match_key", how="inner").with_columns(
    tw_raw=pl.when(pl.col("round") <= 12).then((pl.col("winner") == 2).cast(pl.Int64))
    .otherwise((pl.col("winner") == 3).cast(pl.Int64)),
    flip13=(pl.col("round") == 13) & (pl.col("grp") == "lateA"))
RR = RR.with_columns(tw_flip=pl.when(pl.col("flip13")).then(1 - pl.col("tw_raw")).otherwise(pl.col("tw_raw")))
T = RR.group_by("match_key").agg(pl.col("tw_raw").sum().alias("W_T_raw"), pl.col("tw_flip").sum().alias("W_T_flip13"),
                                 pl.len().alias("n")).join(C, on="match_key", how="left")
T = T.with_columns(ok_raw=pl.col("W_T_raw") == pl.col("t_starters_score_final"),
                   ok_flip=pl.col("W_T_flip13") == pl.col("t_starters_score_final"),
                   grp=pl.col("match_key").str.split("/").list.get(3).cast(pl.String))
print("\n== team-identity totals vs header final score (matches with 24 contiguous rounds, sum consistent)")
print("   regime, n, raw-label agrees, flip-13 agrees:")
regime = M.select(["match_key", "regime", "day"])
T = T.join(regime, on="match_key", how="left")
print(T.group_by("regime").agg(pl.len().alias("n"), pl.col("ok_raw").sum().alias("raw_ok"),
                              pl.col("ok_flip").sum().alias("flip13_ok")).sort("regime").rows())
print("   by day class:")
T = T.with_columns(dclass=pl.when(pl.col("day") >= "12").then(pl.lit("late")).otherwise(pl.lit("early")))
print(T.group_by(["dclass", "regime"]).agg(pl.len().alias("n"), pl.col("ok_raw").sum().alias("raw_ok"),
                                          pl.col("ok_flip").sum().alias("flip13_ok")).sort(["dclass", "regime"]).rows())
