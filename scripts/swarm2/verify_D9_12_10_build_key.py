"""verify D9-12 step 10: per-build check of the 2024 round_start offset (score anchor by build_num).

Header rows carry build_num and a key equal to match_key (csds/YYYY/MM/DD/<uuid>/csds). For each 2024 round_start
row, the score-sum anchor gives dev = (cumulative score sum at the row tick) - (label - 1). dev = 1 means the row
labelled r starts physical round r+1. Grouped by build_num. Read-only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    fs = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    lf = pl.scan_parquet(fs)
    have = set(lf.collect_schema().names())
    return lf.select([c for c in cols if c in have]).collect()


for year, (base, suf) in TOMES.items():
    print(f"\n################ {year}")
    hdr_fs = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    H = pl.scan_parquet(hdr_fs).select(["key", "build_num"]).unique(subset=["key"]).collect()
    print("header rows:", H.height, "| build_num counts:", H.group_by("build_num").len().sort("build_num").rows())
    rs = scan(base, suf, "round_start", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("tick").min().alias("rs"))
    st = scan(base, suf, "round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
    sc = (st.filter(pl.col("t_score").is_not_null() & pl.col("ct_score").is_not_null())
          .with_columns((pl.col("t_score") + pl.col("ct_score")).alias("ssum"))
          .select(["match_key", "tick", "ssum"]).sort("tick"))
    left = rs.sort("rs").with_columns(pl.col("rs").alias("_t"))
    anc = left.join_asof(sc.rename({"tick": "_t"}), on="_t", by="match_key", strategy="backward")
    anc = anc.join(H.rename({"key": "match_key"}), on="match_key", how="left")
    anc = anc.with_columns((pl.col("ssum") - (pl.col("round") - 1)).alias("dev"))
    print("rows with a build match:", anc.filter(pl.col("build_num").is_not_null()).height, "of", anc.height)
    print("dev (= sum - (label-1)) by build_num:", anc.group_by(["build_num", "dev"]).len().sort(["build_num", "dev"]).rows())
    # per-match single-offset check within each build
    pm = anc.group_by("match_key").agg(pl.col("dev").n_unique().alias("nd"), pl.col("build_num").first())
    print("matches with more than one dev value:", pm.filter(pl.col("nd") > 1).height, "of", pm.height)
