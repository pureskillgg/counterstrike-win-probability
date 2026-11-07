r"""D5-8 verifier: which rounds are inverted per match? Step-disagreement set vs header totals.

For each contiguous regime-A match with header sum == rounds present:
  D = rounds r where a clean, no-phase-change freeze-end step disagrees with the label (step != winner)
  W_fix = team-A total after flipping exactly the labels in D (MR12 + OT swap model: swaps after 12, 24, 27, 30)
  fit = (W_fix == header T-starter final)
Also: the same count for the raw labels (no flips) and for the claimed set {13, 25, 28} restricted to rounds >= 12.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_multisite.py
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
S_SWAPS = [12, 24, 27, 30, 33]
hfiles = sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))
hdr = pl.concat([pl.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final"])
                 for f in hfiles], how="vertical_relaxed").unique("match_id")
R = pl.read_parquet(OUT / "rounds_v2.parquet").sort(["match_key", "round"])
M = pl.read_parquet(OUT / "matches_v.parquet")

agg = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                 pl.col("round").n_unique().alias("nr"))
agg = agg.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")),
                       mid=pl.col("match_key").str.split("/").list.get(4))
agg = agg.join(hdr, left_on="mid", right_on="match_id", how="left").with_columns(
    fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).join(
    M.select(["match_key", "regime"]), on="match_key", how="left")
ok = agg.filter(pl.col("contig") & (pl.col("fsum") == pl.col("rmax")) & (pl.col("regime") == "A")
                & pl.col("rmax").is_between(18, 30))
keys = ok.select(["match_key", "rmax", "t_starters_score_final"])

# clean no-phase-change steps for each round
X = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
X = X.filter((pl.col("nr") == pl.col("round") + 1) & (pl.col("fe_phase") == pl.col("nph"))
             & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
X = X.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2).when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3)
    .otherwise(None).cast(pl.Int64))
X = X.filter(pl.col("st").is_not_null()).select(["match_key", "round", "st", "winner"]).join(keys, on="match_key", how="inner")
D = X.filter(pl.col("st") != pl.col("winner")).select(["match_key", "round"]).unique()
Dset = D.group_by("match_key").agg(pl.col("round").sort().alias("D"))

RR = R.join(keys, on="match_key", how="inner").join(Dset, on="match_key", how="left").with_columns(
    D=pl.col("D").fill_null(pl.lit([], dtype=pl.List(pl.Int64))))
RR = RR.with_columns(nsw=sum([(pl.col("round") > b).cast(pl.Int64) for b in S_SWAPS]))
RR = RR.with_columns(sideA=pl.when(pl.col("nsw") % 2 == 0).then(2).otherwise(3))
RR = RR.with_columns(w_raw=(pl.col("winner") == pl.col("sideA")).cast(pl.Int64),
                     inD=pl.col("D").list.contains(pl.col("round")).fill_null(False))
RR = RR.with_columns(w_fix=pl.when(pl.col("inD")).then(1 - pl.col("w_raw")).otherwise(pl.col("w_raw")),
                     w_claim=pl.when(pl.col("round").is_in([13, 25, 28])).then(1 - pl.col("w_raw")).otherwise(pl.col("w_raw")))
T = RR.group_by("match_key").agg(pl.col("w_raw").sum().alias("A_raw"), pl.col("w_fix").sum().alias("A_fix"),
                                 pl.col("w_claim").sum().alias("A_claim"), pl.col("t_starters_score_final").first().alias("t"),
                                 pl.col("rmax").first().alias("rmax"), pl.col("D").first().alias("D"))
T = T.with_columns(Dstr=pl.col("D").cast(pl.List(pl.String)).list.join(","))
print("matches:", T.height, " by rmax:", T.group_by("rmax").len().sort("rmax").rows())
print("exact fit: raw", int((T["A_raw"] == T["t"]).sum()), " flip-D-set", int((T["A_fix"] == T["t"]).sum()),
      " claimed {13,25,28}", int((T["A_claim"] == T["t"]).sum()))
print("\nD-set patterns (rounds with clean step != label), counts, and whether flip-D fits the header:")
pat = T.group_by("Dstr").agg(pl.len().alias("n"), (pl.col("A_fix") == pl.col("t")).sum().alias("fix_fit")).sort("n", descending=True)
print(pat.head(25).rows())
print("\nround-9 membership and 13 membership (per match), header fit of flip-D:")
T = T.with_columns(has9=pl.col("D").list.contains(9).fill_null(False), has13=pl.col("D").list.contains(13).fill_null(False))
print(T.group_by(["has9", "has13"]).agg(pl.len().alias("n"), (pl.col("A_fix") == pl.col("t")).sum().alias("fix_fit"),
                                       (pl.col("A_raw") == pl.col("t")).sum().alias("raw_fit")).sort(["has9", "has13"]).rows())
