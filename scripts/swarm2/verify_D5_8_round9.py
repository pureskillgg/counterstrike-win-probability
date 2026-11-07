r"""D5-8 verifier: where do the round-9 step disagreements in regime A live? (contiguity, header status, rmax, halftime)
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_round9.py
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
OUT = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")
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
    M.select(["match_key", "regime", "day"]), on="match_key", how="left")
agg = agg.with_columns(sumok=pl.col("fsum") == pl.col("rmax"))
X = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
X = X.filter((pl.col("nr") == pl.col("round") + 1) & (pl.col("fe_phase") == pl.col("nph"))
             & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
X = X.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2).when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3)
    .otherwise(None).cast(pl.Int64)).filter(pl.col("st").is_not_null())
bad9 = X.filter((pl.col("round") == 9) & (pl.col("st") != pl.col("winner"))).select("match_key").unique()
A = agg.filter(pl.col("regime") == "A").with_columns(has9=pl.col("match_key").is_in(bad9["match_key"].implode()))
print("regime A matches:", A.height, " with round-9 step disagreement:", int(A.filter("has9").height) if False else int(A.filter(pl.col("has9")).height))
print("by contig / sumok:", A.group_by(["contig", "sumok", "has9"]).len().sort(["contig", "sumok", "has9"]).rows())
print("rmax distribution among has9:", A.filter(pl.col("has9")).group_by("rmax").len().sort("rmax").head(30).rows())
print("rmin distribution among has9:", A.filter(pl.col("has9")).group_by("rmin").len().sort("rmin").rows())
print("day distribution among has9 (late):", A.filter(pl.col("has9")).group_by("day").len().sort("day").rows())
