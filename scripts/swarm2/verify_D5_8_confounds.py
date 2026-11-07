r"""D5-8 verifier: confound checks.

(1) day x regime: bomb-side mismatch at round 13 by day and regime (is regime, or date, the driver?)
(2) regime B completeness: min round and gap structure per match
(3) header-fit by halftime position for short contiguous regime-A matches (rmax 13..17): which swap set fits?
(4) own score-step measure for the claim's 13 -> 14 boundary: label vs fe-derived step, regime A vs B (no swap at 13->14)
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_confounds.py
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
R = pl.read_parquet(OUT / "rounds_v2.parquet").sort(["match_key", "round"])
M = pl.read_parquet(OUT / "matches_v.parquet")
hfiles = sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))
import polars as _p
hdr = _p.concat([_p.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final"])
                 for f in hfiles], how="vertical_relaxed").unique("match_id")

print("== (1) day x regime: bomb-side mismatch at round 13 (rounds with a bomb outcome)")
b13 = R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null()).with_columns(
    mm=(pl.col("winner") != pl.col("bside")).cast(pl.Int64))
t1 = b13.group_by(["day", "regime"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["day", "regime"])
print(t1.rows())
print("   regime A days:", M.filter(pl.col("regime") == "A").group_by("day").len().sort("day").rows())
print("   regime B days:", M.filter(pl.col("regime") == "B").group_by("day").len().sort("day").rows())

print("\n== (2) regime B completeness: min round and number of missing rounds (rmax-based)")
g = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                               pl.col("round").n_unique().alias("nr")).join(M.select(["match_key", "regime", "day"]),
                                                                            on="match_key")
g = g.with_columns(missing=pl.col("rmax") - pl.col("nr") + 0)
print(g.group_by(["regime"]).agg(pl.len().alias("matches"), (pl.col("rmin") > 1).sum().alias("rmin_gt1"),
                                 (pl.col("missing") > 0).sum().alias("with_gaps"),
                                 pl.col("missing").mean().round(2).alias("mean_missing"),
                                 pl.col("nr").mean().round(2).alias("mean_rounds")).sort("regime").rows())

print("\n== (3) short contiguous regime-A matches: which halftime position fits the header T-starter total?")
agg = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                 pl.col("round").n_unique().alias("nr"))
agg = agg.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")),
                       mid=pl.col("match_key").str.split("/").list.get(4))
agg = agg.join(hdr, left_on="mid", right_on="match_id", how="left").with_columns(
    fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
S = agg.filter(pl.col("contig") & (pl.col("fsum") == pl.col("rmax")) & pl.col("rmax").is_between(13, 17))
print("   matches:", S.height, " by rmax:", S.group_by("rmax").len().sort("rmax").rows())
RS = R.join(S.select(["match_key", "rmax", "t_starters_score_final"]), on="match_key", how="inner")
res = []
for H in range(6, 16):
    for flip in [None, 13]:
        # team A = starter T; swap after H; flip optional
        d = RS.with_columns(nsw=(pl.col("round") > H).cast(pl.Int64),
                            sideA=pl.when(((pl.col("round") > H).cast(pl.Int64)) % 2 == 0).then(2).otherwise(3))
        d = d.with_columns(w=(pl.col("winner") == pl.col("sideA")).cast(pl.Int64))
        if flip is not None:
            d = d.with_columns(w=pl.when(pl.col("round") == flip).then(1 - pl.col("w")).otherwise(pl.col("w")))
        tot = d.group_by("match_key").agg(pl.col("w").sum().alias("A"), pl.col("t_starters_score_final").first().alias("t"))
        fit = int((tot["A"] == tot["t"]).sum())
        res.append((H, flip, fit, tot.height))
print("   (halftime after round H, flip round, exact fits, matches):")
for r in res:
    print("   ", r)

print("\n== (4) claim-style score-step at 13 -> 14 (fe_t/fe_ct of 14 minus 13), no phase change, clean +1 step")
A = R.with_columns(nt=pl.col("fe_t").shift(-1).over("match_key"), nct=pl.col("fe_ct").shift(-1).over("match_key"),
                   nr=pl.col("round").shift(-1).over("match_key"), nph=pl.col("fe_phase").shift(-1).over("match_key"))
A = A.filter((pl.col("round") == 13) & (pl.col("nr") == 14) & pl.col("fe_t").is_not_null() & pl.col("nt").is_not_null())
A = A.with_columns(dt=pl.col("nt") - pl.col("fe_t"), dct=pl.col("nct") - pl.col("fe_ct")).with_columns(
    st=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2).when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3)
    .otherwise(None).cast(pl.Int64))
A = A.with_columns(sameph=pl.col("fe_phase") == pl.col("nph"))
print(A.group_by(["grp", "sameph"]).agg(pl.len().alias("rows"), pl.col("st").is_not_null().sum().alias("clean_step"),
                                        (pl.col("st").is_not_null() & (pl.col("st") != pl.col("winner"))).sum().alias("step_ne_label"),
                                        (pl.col("st").is_not_null() & (pl.col("st") == pl.col("winner"))).sum().alias("step_eq_label")
                                        ).sort(["grp", "sameph"]).rows())
