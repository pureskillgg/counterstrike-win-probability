r"""D5-8 verifier: overlap of round-9 step disagreements with header-consistent short matches, and a per-match
step-disagreement-set fit for the 13..17 round group (which the flip-13 variant did not fit).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_shortfit.py
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
agg = (agg.join(hdr, left_on="mid", right_on="match_id", how="left")
       .with_columns(fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
       .join(M.select(["match_key", "regime"]), on="match_key", how="left")
       .with_columns(sumok=pl.col("fsum") == pl.col("rmax")))
A = agg.filter(pl.col("regime") == "A")

X = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
X = X.filter((pl.col("nr") == pl.col("round") + 1) & (pl.col("fe_phase") == pl.col("nph"))
             & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
X = X.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2)
    .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3).otherwise(None).cast(pl.Int64))
X = X.filter(pl.col("st").is_not_null())
Dd = X.filter(pl.col("st") != pl.col("winner")).select(["match_key", "round"]).unique()
Dset = Dd.group_by("match_key").agg(pl.col("round").sort().alias("D"))

# (a) overlap of round-9 disagreements with header-consistent short matches
print("== (a) regime A matches with round-9 disagreement, by rmax and header status")
b9 = Dd.filter(pl.col("round") == 9).select("match_key").unique()
A9 = A.with_columns(h9=pl.col("match_key").is_in(b9["match_key"].implode()))
print(A9.filter(pl.col("h9")).group_by(["rmax", "contig", "sumok"]).len().sort(["rmax", "contig", "sumok"]).rows())

# (b) per-match step-disagreement set fit for short contiguous header-consistent matches, rmax 10..17
print("\n== (b) header fit when flipping exactly the step-disagreement set D, rmax 10..17 (contig, sumok)")
sh = A.filter(pl.col("contig") & pl.col("sumok") & pl.col("rmax").is_between(10, 17)).select(
    ["match_key", "rmax", "t_starters_score_final"]).join(Dset, on="match_key", how="left").with_columns(
    D=pl.col("D").fill_null(pl.lit([], dtype=pl.List(pl.Int64))))
RS = R.join(sh.select(["match_key", "rmax", "t_starters_score_final", "D"]), on="match_key", how="inner")
RS = RS.with_columns(nsw=sum([(pl.col("round") > b).cast(pl.Int64) for b in S_SWAPS]))
RS = RS.with_columns(sideA=pl.when(pl.col("nsw") % 2 == 0).then(2).otherwise(3))
RS = RS.with_columns(w=(pl.col("winner") == pl.col("sideA")).cast(pl.Int64),
                     inD=pl.col("D").list.contains(pl.col("round")).fill_null(False))
RS = RS.with_columns(wd=pl.when(pl.col("inD")).then(1 - pl.col("w")).otherwise(pl.col("w")))
T = RS.group_by("match_key").agg(pl.col("w").sum().alias("raw"), pl.col("wd").sum().alias("fixD"),
                                 pl.col("t_starters_score_final").first().alias("t"), pl.col("rmax").first().alias("rmax"),
                                 pl.col("D").first().alias("D"))
T = T.with_columns(Dstr=pl.col("D").cast(pl.List(pl.String)).list.join(",").fill_null("none"))
print(T.group_by(["rmax"]).agg(pl.len().alias("n"), (pl.col("raw") == pl.col("t")).sum().alias("raw_fit"),
                               (pl.col("fixD") == pl.col("t")).sum().alias("fixD_fit")).sort("rmax").rows())
print("pattern (Dstr, n, raw_fit, fixD_fit):",
      T.group_by("Dstr").agg(pl.len().alias("n"), (pl.col("raw") == pl.col("t")).sum().alias("raw_fit"),
                             (pl.col("fixD") == pl.col("t")).sum().alias("fixD_fit")).sort("n", descending=True).head(15).rows())
