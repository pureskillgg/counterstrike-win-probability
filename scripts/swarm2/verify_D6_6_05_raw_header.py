"""D6-6 verifier step 05: source-level check of the build boundary (read-only on E:).

  (a) timestamp precision: matches whose header match_date has nanosecond precision (non-.000Z), by build and day.
  (b) the single 9842 match dated 2024-02-12T03:55:50 (between two 9961 matches): its round-13 status.
  (c) raw E: header channel for a sample of build 9842 and build 9961 matches: does the raw header carry the same
      build_num as the tome, so the build split is in the source demos and not added by the tome build?
  (d) round 25 and round 28 scorability and disagreement under the D6 score rule, by build (reconciles D6-4's
      round-25 figure, 124 of 178, with this recompute).
Reads E: only (no writes). Tome reads are column-projected.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_05_raw_header.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(os.environ.get("VERIFY_SCRATCH", r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"))
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
RAW24 = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")

H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "game_directory"])
               for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))], how="vertical_relaxed")
H = H.unique(["key"], keep="first", maintain_order=True).with_columns(
    pl.col("match_date").alias("ts"),
    pl.col("match_date").str.slice(0, 10).alias("md"),
    pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("key_day"),
    pl.col("key").str.split("/").list.get(4).alias("uuid"),
    (~pl.col("match_date").str.contains(r"\.000Z$")).alias("ns_ts"),
)

print("== (a) timestamp precision by build (2024) ==")
print(H.group_by(["build_num", "ns_ts"]).len().sort(["build_num", "ns_ts"]))
ns = H.filter(pl.col("ns_ts"))
print("ns-precision matches by md (first 30):")
print(ns.group_by(["md", "build_num"]).len().sort("md").head(30).rows())
print("ns-precision matches by key_day:", ns.group_by("key_day").len().sort("key_day").rows())

L = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet").join(
    H.select(["key", "build_num", "ts", "ns_ts", "md"]).rename({"key": "match"}), on="match", how="left", suffix="_h")
r13 = L.filter((pl.col("round") == 13) & pl.col("scorable"))
print("\nround-13 scorable by build and ns_ts:")
print(r13.group_by(["build_num", "ns_ts"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort(["build_num", "ns_ts"]))

print("\n== (b) the 9842 match dated after the cut (2024-02-12, nanosecond stamp) ==")
odd = H.filter((pl.col("build_num") == 9842) & (pl.col("md") >= "2024-02-08"))
print(odd.select(["key", "match_date", "build_num", "game_directory", "key_day"]).rows())
for k in odd["key"].to_list():
    print("  round-13 row for", k[:40], ":", L.filter((pl.col("match") == k) & (pl.col("round") == 13))
          .select(["round", "w_sc", "w_re", "bomb", "scorable", "sc_dis"]).rows())

print("\n== (c) raw E: header channel vs tome build_num (sample) ==")
rng = H.filter(pl.col("md") >= "2024-02-01").sample(n=12, seed=7)  # sample across 9842/9961, plus a few
samp = pl.concat([
    H.filter((pl.col("build_num") == 9842) & (pl.col("md") < "2024-02-07")).sample(n=5, seed=11),
    H.filter((pl.col("build_num") == 9961) & (pl.col("md") >= "2024-02-07")).sample(n=5, seed=13),
    odd.head(2),
])
for row in samp.iter_rows(named=True):
    folder = RAW24 / row["key_day"] / row["uuid"]
    chans = sorted(p.name for p in folder.iterdir()) if folder.exists() else []
    info = {"key_day": row["key_day"], "tome_build": row["build_num"], "tome_ts": row["ts"], "raw_channels": len(chans),
            "has_header": "header" in chans}
    if "header" in chans:
        hd = pl.read_parquet(str(folder / "header"))
        cols = hd.columns
        info["raw_build_num"] = hd["build_num"][0] if "build_num" in cols else "no_col"
        info["raw_match_date"] = hd["match_date"][0] if "match_date" in cols else "no_col"
    print(info)

print("\nraw channel names for one 9961 sample (schema check):")
for row in samp.filter(pl.col("build_num") == 9961).head(1).iter_rows(named=True):
    folder = RAW24 / row["key_day"] / row["uuid"]
    print(sorted(p.name for p in folder.iterdir()) if folder.exists() else "missing folder")

print("\n== (d) rounds 25 and 28: scorability and disagreement by build (D6 rule) ==")
for rr in (9, 13, 25, 27, 28):
    sub = L.filter(pl.col("round") == rr)
    print(f"round {rr}: rows with round_end={sub.filter(pl.col('w_re').is_not_null()).height}; "
          f"score winner present={sub.filter(pl.col('w_sc').is_not_null()).height}; scorable={sub.filter(pl.col('scorable')).height}")
    print("   by build (scorable n, disagree):",
          sub.filter(pl.col("scorable")).group_by("build_num").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("build_num").rows())
