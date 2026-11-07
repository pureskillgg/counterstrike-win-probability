"""D9-12 verifier, extra cross-check (outside the D9-12 claim itself; read-only on the tomes).

Physical winner of round r from scores: between the round_freeze_end rows of labels r and r+1, the T-side score
increments (dT=1, dCT=0) means side T (code 2) won round r; a CT-side increment (dCT=1, dT=0) means side CT (code 3).
This uses no bomb, death or round_end winner field. Compared with round_end winner_team_code of label r.
Only labels r with freeze rows for both r and r+1 are used (ASSUMPTION W: score deltas reflect round r only).
Reported by build and by round index (13, 25, 28 and 9 are the D9 indices; others pooled).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}
KEY = ["match_key", "round"]


def scan(base, suf, ch, cols):
    files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


for tome, (base, suf) in TOMES.items():
    print(f"\n########## tome {tome}")
    H = pl.concat([pl.read_parquet(f, columns=["key", "build_num"]) for f in
                   sorted(glob.glob(str(base / "header" / "dataframe_*")))]).rename({"key": "match_key"})
    RS = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    FE = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick") \
        .group_by(KEY).agg(pl.col("t_score").first(), pl.col("ct_score").first())
    RE = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code"]).sort("tick") \
        .group_by(KEY).agg(pl.col("winner_team_code").first())
    nxt = FE.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t_next", "ct_score": "ct_next"})
    W = FE.join(nxt, on=KEY, how="inner")  # label r freeze + label r+1 freeze
    W = W.with_columns((pl.col("t_next") - pl.col("t_score")).alias("dT"),
                       (pl.col("ct_next") - pl.col("ct_score")).alias("dCT"))
    W = W.with_columns(
        pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
        .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3)
        .otherwise(None).alias("score_winner"))
    W = W.join(RE, on=KEY, how="inner").join(H, on="match_key", how="left")
    W = W.with_columns(
        pl.when(pl.col("score_winner").is_null()).then(pl.lit("no clean score delta"))
        .when(pl.col("score_winner") == pl.col("winner_team_code")).then(pl.lit("agree"))
        .otherwise(pl.lit("disagree")).alias("status"))
    print(f"   rounds with freeze(r) and freeze(r+1) and a round_end row: {W.height}")
    print("   status overall:", W.group_by("status").len().sort("status").rows())
    W = W.with_columns(
        pl.when(pl.col("round").is_in([9, 13, 25, 28])).then(pl.col("round").cast(pl.String))
        .otherwise(pl.lit("other")).alias("rclass"))
    print("   status by build and round class:")
    for row in W.group_by(["build_num", "rclass", "status"]).len().sort(["build_num", "rclass", "status"]).rows():
        print("      ", row)
    r13 = W.filter((pl.col("round") == 13)).group_by(["build_num", "status"]).len().sort(["build_num", "status"])
    print("   round 13 only:", r13.rows())
    print("   score deltas seen (dT, dCT) counts:", W.group_by(["dT", "dCT"]).len().sort(["dT", "dCT"]).rows())
