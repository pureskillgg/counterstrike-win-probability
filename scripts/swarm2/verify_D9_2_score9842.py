"""verify_D9_2 score9842: header-score consistency for build 9842 under recorded vs round-13-flipped codes.

Build 9842 round_end labels start at 2 (round 1 absent). For matches whose header total exceeds the label count by
exactly 1, the single missing round is taken as round 1 (ASSUMPTION). Then, with d_T = T_hdr - T_k and
d_C = CT_hdr - C_k, the missing round adds 1 to exactly one side: (d_T, d_C) in {(1,0),(0,1)}.
Same test for build 9961 (round 1 present) on matches with header total exceeding the label count by exactly 1,
where the missing round is assumed to be the last round (ASSUMPTION), so (d_T, d_C) in {(1,0),(0,1)} is again
the consistency condition for recorded codes.
Reports consistency rates under recorded codes and under round-13-flipped codes.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")

R = pl.read_parquet(SCR / "verify_D9_2_round.parquet").select(
    ["match_key", "round", "winner_team_code", "build_num", "has_out"])
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final"]).collect().with_columns(
    pl.col("key").alias("match_key"))
pm = R.group_by("match_key").agg(pl.len().alias("n_lab"), pl.col("build_num").first()).join(
    H.select(["match_key", "t_starters_score_final", "ct_starters_score_final"]), on="match_key")
pm = pm.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")
                      - pl.col("n_lab")).alias("gap"))
print("gap = header total - label count, by build (value: count):")
print(pm.group_by(["build_num", "gap"]).len().sort(["build_num", "gap"]).head(20).rows())


def consistency(build, gap_value, flip13):
    sub = R.join(pm.filter((pl.col("build_num") == build) & (pl.col("gap") == gap_value)).select("match_key"),
                 on="match_key", how="inner")
    if sub.height == 0:
        return None
    w = pl.col("winner_team_code")
    if flip13:
        w = pl.when(pl.col("round") == 13).then(pl.when(w == 2).then(3).otherwise(2)).otherwise(w)
    d = sub.with_columns(w.alias("w")).with_columns(
        pl.when((pl.col("round") <= 12) & (pl.col("w") == 2)).then(1).otherwise(0).alias("a"),
        pl.when((pl.col("round") >= 13) & (pl.col("w") == 3)).then(1).otherwise(0).alias("b"),
        pl.when((pl.col("round") <= 12) & (pl.col("w") == 3)).then(1).otherwise(0).alias("c"),
        pl.when((pl.col("round") >= 13) & (pl.col("w") == 2)).then(1).otherwise(0).alias("d")).group_by(
        "match_key").agg((pl.col("a") + pl.col("b")).sum().alias("T"), (pl.col("c") + pl.col("d")).sum().alias("C"))
    d = d.join(pm.select(["match_key", "t_starters_score_final", "ct_starters_score_final"]), on="match_key")
    d = d.with_columns((pl.col("t_starters_score_final") - pl.col("T")).alias("dT"),
                       (pl.col("ct_starters_score_final") - pl.col("C")).alias("dC"))
    # missing rounds add `gap` wins split between the two starting sides: any nonnegative split summing to gap
    ok = (pl.col("dT") >= 0) & (pl.col("dC") >= 0) & ((pl.col("dT") + pl.col("dC")) == gap_value)
    n = d.height
    k = int(d.filter(ok).height)
    return n, k


for build, gap_value in [(9842, 1), (9842, 2), (9961, 1)]:
    rec = consistency(build, gap_value, flip13=False)
    flp = consistency(build, gap_value, flip13=True)
    print(f"build {build}, gap {gap_value}: recorded consistent {rec[1]}/{rec[0]} = {rec[1]/rec[0]:.3f} | "
          f"round-13 flipped consistent {flp[1]}/{flp[0]} = {flp[1]/flp[0]:.3f}")
print("\nDONE score9842")
