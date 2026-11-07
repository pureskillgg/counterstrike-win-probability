"""D6-6 verifier step 03: is there a header build/version change at the 2024 match-date cut, and where exactly
does the round-13 defect start in time (UTC timestamp)? Also checks whether 2026 folder day equals match day.

Uses the long tables written by verify_D6_6_02_recompute.py (scratchpad) and header columns from the tome.
Read-only. ASSUMPTION A1 (UTC) is tested by printing the exact transition timestamp.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_03_build_cut.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(os.environ.get("VERIFY_SCRATCH", r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"))
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
BASE26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"

HCOLS = ["key", "match_date", "map_name", "build_num", "demo_version_name", "demo_version_guid",
         "network_protocol", "server_name", "game_directory", "fullpackets_version", "tick_rate"]
H24 = pl.concat([pl.read_parquet(f, columns=HCOLS) for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))],
                how="vertical_relaxed")
H24 = H24.with_columns(pl.col("key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("key_day"),
                       pl.col("match_date").str.slice(0, 10).alias("md"),
                       pl.col("match_date").alias("ts"))

print("== A. header build/version fields, 2024: distinct values ==")
for c in ["build_num", "demo_version_name", "demo_version_guid", "network_protocol", "server_name", "game_directory",
          "fullpackets_version", "tick_rate"]:
    vc = H24.group_by(c).len().sort("len", descending=True)
    print(f"{c}: distinct={vc.height} top={vc.head(4).rows()}")

print("\n== B. build_num by match date (UTC) around the cut ==")
w = H24.filter((pl.col("md") >= "2024-02-03") & (pl.col("md") <= "2024-02-10"))
print(w.group_by(["md", "build_num"]).len().sort(["md", "build_num"]))
print("\nbuild_num by key folder day (2024-02 folders 06..13 only):")
print(H24.filter(pl.col("key_day").is_in(["06", "07", "12", "13"])).group_by(["key_day", "build_num"]).len().sort(["key_day", "build_num"]))

print("\n== C. exact transition timestamp for round-13 score disagreement (scorable rows) ==")
L24 = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet")
r13 = L24.filter((pl.col("round") == 13) & pl.col("scorable")).select(["match", "sc_dis", "md", "key_day"])
r13 = r13.join(H24.select(["key", "ts", "build_num"]).rename({"key": "match"}), on="match", how="left")
r13s = r13.filter(pl.col("md").is_in(["2024-02-06", "2024-02-07"])).sort("ts")
print(r13s.select(["ts", "md", "key_day", "sc_dis", "build_num"]).rows())
clean_max = r13.filter(~pl.col("sc_dis")).select(pl.col("ts").max()).item()
dis_min = r13.filter(pl.col("sc_dis")).select(pl.col("ts").min()).item()
print("latest clean (score==round_end) round-13 match timestamp:", clean_max)
print("earliest disagreeing round-13 match timestamp:          ", dis_min)

print("\n== D. sensitivity: round-13 score disagreement by UTC cut (2024-02-07T00:00Z) and by hour cut near it ==")
mid = r13.filter(pl.col("ts").str.slice(0, 13).is_in(["2024-02-06T1", "2024-02-06T2", "2024-02-07T0", "2024-02-07T1"]))
print(mid.with_columns(pl.col("ts").str.slice(0, 13).alias("hour")).group_by("hour").agg(
    pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("hour"))

print("\n== E. 2026: folder day vs match day ==")
H26 = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "max_rounds"]) for f in sorted(glob.glob(str(BASE26 / "header" / "dataframe_*")))],
                how="vertical_relaxed")
H26 = H26.with_columns(pl.col("key").str.extract(r"csds/2026/10/(\d\d)/", 1).alias("key_day"),
                       pl.col("match_date").str.slice(0, 10).alias("md"))
print("2026 header rows:", H26.height, " key_day null:", H26.filter(pl.col("key_day").is_null()).height)
print("2026 key_day == md day-of-month:", H26.filter(pl.col("key_day") == pl.col("md").str.slice(8, 2)).height, "of", H26.height)
print(H26.group_by(["key_day"]).len().sort("key_day").rows())
print("2026 md range:", H26["md"].min(), H26["md"].max())

print("\n== F. 2024 match counts per key_day vs md day, folder 12 composition ==")
f12 = H24.filter(pl.col("key_day") == "12")
print("folder 12 rows:", f12.height, " md < 2024-02-07:", f12.filter(pl.col("md") < "2024-02-07").height,
      " md >= 2024-02-07:", f12.filter(pl.col("md") >= "2024-02-07").height)
print("folder 12 md distribution (2024-02-07..2024-02-12):",
      f12.filter(pl.col("md") >= "2024-02-07").group_by("md").len().sort("md").rows())
