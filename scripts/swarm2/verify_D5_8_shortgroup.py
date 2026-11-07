r"""D5-8 verifier: the short contiguous regime-A group (rmax 13..17) that fits neither label variant.

Checks: (a) complement mapping (header t vs team that started CT), (b) presence of surrender reasons (17, 18),
(c) regime B first-round distribution (rmin) to confirm the round-1 gap structure.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_shortgroup.py
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
hfiles = sorted(__import__("glob").glob(os.path.join(BASE, "header", "dataframe_*")))
hdr = pl.concat([pl.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final"])
                 for f in hfiles], how="vertical_relaxed").unique("match_id")
R = pl.read_parquet(OUT / "rounds_v2.parquet").sort(["match_key", "round"])
M = pl.read_parquet(OUT / "matches_v.parquet")

print("== (c) regime B first round (rmin) and regime A:")
g = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                               pl.col("round").n_unique().alias("nr")).join(M.select(["match_key", "regime"]),
                                                                            on="match_key")
print(g.group_by(["regime", "rmin"]).len().sort(["regime", "rmin"]).head(20).rows())

agg = g.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")),
                     mid=pl.col("match_key").str.split("/").list.get(4))
agg = agg.join(hdr, left_on="mid", right_on="match_id", how="left").with_columns(
    fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
S = agg.filter(pl.col("contig") & (pl.col("fsum") == pl.col("rmax")) & pl.col("rmax").is_between(13, 17))
RS = R.join(S.select(["match_key", "rmax", "t_starters_score_final", "ct_starters_score_final"]), on="match_key",
            how="inner")
print("\n== (a) short group: team-A (starter T) totals under label with halftime at 12, flip13 variant")
d = RS.with_columns(sideA=pl.when((pl.col("round") > 12).cast(pl.Int64) == 0).then(2).otherwise(3))
d = d.with_columns(w=(pl.col("winner") == pl.col("sideA")).cast(pl.Int64))
d = d.with_columns(wf=pl.when(pl.col("round") == 13).then(1 - pl.col("w")).otherwise(pl.col("w")))
tot = d.group_by("match_key").agg(pl.col("w").sum().alias("A_raw"), pl.col("wf").sum().alias("A_f13"),
                                  pl.col("t_starters_score_final").first().alias("t"),
                                  pl.col("ct_starters_score_final").first().alias("ct"),
                                  pl.col("rmax").first().alias("rmax"))
print("   n:", tot.height)
print("   A_raw == t:", int((tot["A_raw"] == tot["t"]).sum()), "  A_raw == ct:", int((tot["A_raw"] == tot["ct"]).sum()))
print("   A_f13 == t:", int((tot["A_f13"] == tot["t"]).sum()), "  A_f13 == ct:", int((tot["A_f13"] == tot["ct"]).sum()))
print("   distribution of (t, ct) pairs:", tot.group_by(["t", "ct"]).len().sort("len", descending=True).head(12).rows())
print("   distribution of (A_raw, A_f13):", tot.group_by(["A_raw", "A_f13"]).len().sort("len", descending=True).head(12).rows())

print("\n== (b) surrender reasons (17/18) in short group vs other regime-A contiguous matches:")
sur = R.filter(pl.col("reason").is_in([17, 18])).select("match_key").unique()
S2 = S.with_columns(has_surr=pl.col("match_key").is_in(sur["match_key"]))
print("   short group with surrender code:", int(S2.filter(pl.col("has_surr")).height), "of", S2.height)
oth = agg.filter(pl.col("contig") & (pl.col("fsum") == pl.col("rmax")) & (pl.col("rmax") >= 18) & (pl.col("rmax") <= 24))
oth = oth.with_columns(has_surr=pl.col("match_key").is_in(sur["match_key"]))
print("   18..24 group with surrender code:", int(oth.filter(pl.col("has_surr")).height), "of", oth.height)
