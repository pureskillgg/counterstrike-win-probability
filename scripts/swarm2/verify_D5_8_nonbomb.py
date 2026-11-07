r"""D5-8 verifier: is the round-13 inversion present on rounds WITHOUT a bomb outcome (not selected by bomb mismatch)?
Also: (t, ct) finals for the regulation/OT header groups, and a provenance check (header vs freeze-end-derived final).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_nonbomb.py
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
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")
hfiles = sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))
hdr = pl.concat([pl.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final"])
                 for f in hfiles], how="vertical_relaxed").unique("match_id")
R = pl.read_parquet(OUT / "rounds_v2.parquet").sort(["match_key", "round"])
M = pl.read_parquet(OUT / "matches_v.parquet")

# (i) score step at 13 -> 14 (fe-derived, same phase on both sides), split by bomb outcome at 13
X = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
X = X.filter((pl.col("round") == 13) & (pl.col("nr") == 14) & (pl.col("fe_phase") == pl.col("nph"))
             & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
X = X.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2).when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3)
    .otherwise(None).cast(pl.Int64),
    hasbomb=pl.col("bside").is_not_null())
X = X.filter(pl.col("st").is_not_null())
print("== (i) clean score step at 13->14 vs label at 13, split by bomb outcome at 13")
print(X.group_by(["grp", "hasbomb"]).agg(pl.len().alias("n"), (pl.col("st") != pl.col("winner")).sum().alias("step_ne_label"),
                                         (pl.col("st") == pl.col("winner")).sum().alias("step_eq_label"),
                                         ((pl.col("st") != pl.col("winner")).mean()).round(3).alias("disagree_rate"))
      .sort(["grp", "hasbomb"]).rows())

# (ii) finals by header group, regime A, contiguous, sum-consistent
agg = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                 pl.col("round").n_unique().alias("nr"))
agg = agg.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")),
                       mid=pl.col("match_key").str.split("/").list.get(4))
agg = agg.join(hdr, left_on="mid", right_on="match_id", how="left").with_columns(
    fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).join(
    M.select(["match_key", "regime"]), on="match_key", how="left")
ok = agg.filter(pl.col("contig") & (pl.col("fsum") == pl.col("rmax")) & (pl.col("regime") == "A"))
print("\n== (ii) header (t,ct) finals for contiguous regime-A sum-consistent matches (top groups by rmax):")
for rm in [24, 30, 23, 20]:
    sub = ok.filter(pl.col("rmax") == rm)
    print("   rmax", rm, "n", sub.height, " (t,ct) top:", sub.group_by(["t_starters_score_final", "ct_starters_score_final"]).len().sort(
        "len", descending=True).head(6).rows())

# (iii) provenance: header T-starter final vs freeze-end-derived final, using label at the last round (N != 13)
F = R.filter(pl.col("round") == pl.col("rmax").over("match_key") if False else pl.lit(True))
last = R.group_by("match_key").agg(pl.col("round").max().alias("N"))
L = R.join(last, on="match_key").filter(pl.col("round") == pl.col("N")).select(
    ["match_key", "round", "fe_t", "fe_ct", "winner"]).join(ok.select(["match_key", "t_starters_score_final",
                                                                       "ct_starters_score_final", "rmax"]), on="match_key")
L = L.with_columns(sideA_last=pl.when(pl.col("rmax") <= 12).then(2).otherwise(
    pl.when((pl.col("rmax") > 12) & (pl.col("rmax") <= 24)).then(3).otherwise(pl.lit(None))))
print("\n== (iii) provenance: rounds where last-round freeze-end fields exist (N, per match):",
      L.height)
print("   fe-based final for team A (T-starter) = fe_side(N) + [A wins N] vs header t (N in 19..24 only):")
Lx = L.filter(pl.col("rmax").is_between(19, 24)).with_columns(
    faA=pl.when(pl.col("sideA_last") == 2).then(pl.col("fe_t")).otherwise(pl.col("fe_ct")))
Lx = Lx.with_columns(winA=(pl.col("winner") == pl.col("sideA_last")).cast(pl.Int64)).with_columns(
    fe_final=pl.col("faA") + pl.col("winA"))
print("   n", Lx.height, " fe_final == header t:", int((Lx["fe_final"] == Lx["t_starters_score_final"]).sum()))
