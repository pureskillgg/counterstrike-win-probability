r"""D5-8 verifier: team-identity totals vs header final scores, general side-swap model, sign and variant checks.

Model (ASSUMPTION): sides swap after rounds in S = {12, 24, 27, 30, ...} (MR12 halftime, end of regulation, then each MR3
overtime half). Team A = the team that started on T. side_A(r) = 2 (T) if the number of swaps before round r is even,
else 3 (CT). Team A's total under a labelling L is sum over r of [L_r == side_A(r)].
Compared with header t_starters_score_final for matches whose rounds are contiguous 1..N and N == header sum.
Variants: raw label; flip 13 only; flip 13,25,28 (claimed inverted set).
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_totals.py
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
S_SWAPS = [12, 24, 27, 30, 33]

hfiles = sorted(glob.glob(os.path.join(BASE, "header", "dataframe_*")))
hdr = pl.concat([pl.read_parquet(f, columns=["match_id", "t_starters_score_final", "ct_starters_score_final"])
                 for f in hfiles], how="vertical_relaxed").unique("match_id")

R = pl.read_parquet(OUT / "rounds_v2.parquet")
M = pl.read_parquet(OUT / "matches_v.parquet")
agg = R.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax"),
                                 pl.col("round").n_unique().alias("nr"))
agg = agg.with_columns(contig=(pl.col("rmin") == 1) & (pl.col("nr") == pl.col("rmax")),
                       mid=pl.col("match_key").str.split("/").list.get(4))
agg = agg.join(hdr, left_on="mid", right_on="match_id", how="left").join(M.select(["match_key", "regime", "day"]),
                                                                       on="match_key", how="left")
agg = agg.with_columns(fsum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
agg = agg.with_columns(sumok=pl.col("fsum") == pl.col("rmax"))

print("== completeness by regime (matches):")
print(agg.group_by(["regime", "contig"]).len().sort(["regime", "contig"]).rows())
print("== contiguous + header sum == rmax, by regime and rmax:")
ok = agg.filter(pl.col("contig") & pl.col("sumok"))
print(ok.group_by(["regime"]).len().rows())
print(ok.group_by("rmax").len().sort("rmax").rows())

# per-round label table for those matches
C = ok.select(["match_key", "rmax", "t_starters_score_final", "day", "regime"])
RR = R.join(C.select("match_key"), on="match_key", how="inner").sort(["match_key", "round"])
RR = RR.with_columns(
    nsw=pl.col("round").map_elements(lambda r: 0, return_dtype=pl.Int64))  # placeholder, replaced below
# number of swaps strictly before round r
def swaps_before(r):
    return sum(1 for b in S_SWAPS if b < r)
sw_expr = pl.lit(0)
for b in S_SWAPS:
    sw_expr = sw_expr + (pl.col("round") > b).cast(pl.Int64)
RR = RR.with_columns(nsw=sw_expr)
RR = RR.with_columns(side_A=pl.when(pl.col("nsw") % 2 == 0).then(2).otherwise(3))
RR = RR.with_columns(winA_raw=(pl.col("winner") == pl.col("side_A")).cast(pl.Int64))
FLIP_13 = pl.col("round") == 13
FLIP_CLAIM = pl.col("round").is_in([13, 25, 28])
RR = RR.with_columns(
    winA_f13=pl.when(FLIP_13).then(1 - pl.col("winA_raw")).otherwise(pl.col("winA_raw")),
    winA_fclaim=pl.when(FLIP_CLAIM).then(1 - pl.col("winA_raw")).otherwise(pl.col("winA_raw")),
)
TOT = (RR.group_by("match_key").agg(pl.col("winA_raw").sum().alias("A_raw"),
                                   pl.col("winA_f13").sum().alias("A_f13"),
                                   pl.col("winA_fclaim").sum().alias("A_fclaim"),
                                   pl.len().alias("n"))
       .join(C, on="match_key", how="left"))
TOT = TOT.with_columns(d_raw=pl.col("A_raw") - pl.col("t_starters_score_final"),
                       d_f13=pl.col("A_f13") - pl.col("t_starters_score_final"),
                       d_fclaim=pl.col("A_fclaim") - pl.col("t_starters_score_final"))
print("\n== signed difference (team-A total minus header T-starters final), by regime and rmax:")
print(TOT.group_by(["regime", "rmax"]).agg(pl.len().alias("n"),
                                           (pl.col("d_raw") == 0).sum().alias("raw_eq"),
                                           (pl.col("d_raw") == -1).sum().alias("raw_m1"),
                                           (pl.col("d_raw") == 1).sum().alias("raw_p1"),
                                           (pl.col("d_f13") == 0).sum().alias("f13_eq"),
                                           (pl.col("d_fclaim") == 0).sum().alias("fclaim_eq")
                                           ).sort(["regime", "rmax"]).rows())
print("\n== totals over all contiguous, header-consistent regime A matches:")
print(TOT.select(pl.len().alias("n"), (pl.col("d_raw") == 0).sum().alias("raw_eq"),
                 (pl.col("d_f13") == 0).sum().alias("f13_eq"),
                 (pl.col("d_fclaim") == 0).sum().alias("fclaim_eq")).rows())
TOT.write_parquet(OUT / "totals_v.parquet")
