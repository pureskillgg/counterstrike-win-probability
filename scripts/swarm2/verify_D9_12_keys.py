"""D9-12 independent verifier, key-level counts for the bake-off round_start filter. Read-only on the tomes.

Counts (match, round) keys, unfiltered, in round_end, round_start (basic table) and round_freeze_end, by build.
Shows which labels are missing and where (first or last label of the match), which is what the bake-off's
'no round_start row' filter would act on. Also lists the one 2024 match that has round_end rows but no round_start rows.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
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
    RE = scan(base, suf, "round_end", ["match_key", "round", "tick"]).select(KEY).unique()
    RST = scan(base, suf, "round_start", ["match_key", "round"]).select(KEY).unique()
    RS = scan(base, suf, "round_state", ["match_key", "round", "event_type"])
    FE = RS.filter(pl.col("event_type") == "round_freeze_end").select(KEY).unique()
    print(f"   keys: round_end={RE.height} round_start={RST.height} freeze_end={FE.height}")
    print(f"   round_end keys with a round_start row: {RE.join(RST, on=KEY, how='semi').height}")
    re_no_rst = RE.join(RST, on=KEY, how="anti")
    rst_no_re = RST.join(RE, on=KEY, how="anti")
    fe_no_re = FE.join(RE, on=KEY, how="anti")
    re_no_fe = RE.join(FE, on=KEY, how="anti")
    print(f"   round_end keys WITHOUT round_start row: {re_no_rst.height}")
    print(f"   round_start keys WITHOUT round_end row: {rst_no_re.height}")
    print(f"   freeze_end keys WITHOUT round_end row: {fe_no_re.height} | round_end keys WITHOUT freeze_end: {re_no_fe.height}")

    # label position of the missing keys inside their match
    lab = RE.group_by("match_key").agg(pl.col("round").min().alias("re_min"), pl.col("round").max().alias("re_max"))
    x = re_no_rst.join(lab, on="match_key", how="left").with_columns(
        pl.when(pl.col("round") == pl.col("re_max")).then(pl.lit("last label"))
        .when(pl.col("round") == pl.col("re_min")).then(pl.lit("first label"))
        .otherwise(pl.lit("middle")).alias("pos")).join(H, on="match_key", how="left")
    print("   round_end keys WITHOUT round_start, by build and position:",
          x.group_by(["build_num", "pos"]).len().sort(["build_num", "pos"]).rows())
    y = rst_no_re.join(H, on="match_key", how="left")
    print("   round_start keys WITHOUT round_end, by build and label:",
          y.group_by(["build_num", "round"]).len().sort(["build_num", "round"]).head(12).rows())

    # freeze_end label 1 and round_end label 1 by build
    f1 = FE.filter(pl.col("round") == 1).join(H, on="match_key", how="left")
    e1 = RE.filter(pl.col("round") == 1).join(H, on="match_key", how="left")
    print("   freeze_end label 1 rows by build:", f1.group_by("build_num").len().sort("build_num").rows())
    print("   round_end label 1 rows by build:", e1.group_by("build_num").len().sort("build_num").rows())

    # the match(es) with round_end rows and no round_start rows at all
    m_end = RE.select("match_key").unique()
    m_st = RST.select("match_key").unique()
    orphan = m_end.join(m_st, on="match_key", how="anti").join(H, on="match_key", how="left")
    print("   matches with round_end rows but no round_start rows:", orphan.rows())
    for mk in orphan["match_key"].to_list():
        rr = RE.filter(pl.col("match_key") == mk)["round"].sort().to_list()
        print("     round_end labels for that match:", rr)
