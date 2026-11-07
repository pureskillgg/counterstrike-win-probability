"""D6-6 verifier step 04: defect metrics grouped by header build_num (2024 tome) and by 2026 build_num.

Inputs: long per-round tables from verify_D6_6_02b_day_recompute.py (scratchpad), header build_num from the tome.
Checks:
  (a) build_num is a single change point in time (no interleaving) and the cut time is exact (UTC).
  (b) round-13 score-vs-round_end disagreement, first post-swap (D6 s17 definition), e2 bomb mismatch, by build.
  (c) map mix by build (is map a confound of the build split?).
  (d) 2026 header build_num (is the 2026 control on the same build as the affected 2024 rows?).
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_04_by_build.py
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


def hdr(base, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(base / "header" / "dataframe_*")))],
                     how="vertical_relaxed").unique(["key"], keep="first", maintain_order=True)


H24 = hdr(BASE24, ["key", "match_date", "build_num", "game_directory", "network_protocol", "map_name"])
H24 = H24.with_columns(pl.col("match_date").alias("ts"), pl.col("match_date").str.slice(0, 10).alias("md"))
H24s = H24.sort("ts")
bn = H24s["build_num"].to_list()
changes = [(i, H24s["ts"][i - 1], H24s["ts"][i], bn[i - 1], bn[i]) for i in range(1, len(bn)) if bn[i] != bn[i - 1]]
print("== (a) build_num change points in time order (2024 tome, 4435 matches) ==")
print("number of changes:", len(changes))
for c in changes[:10]:
    print("   idx", c[0], "last before:", c[1], c[3], " first after:", c[2], c[4])
b9842_max = H24.filter(pl.col("build_num") == 9842)["ts"].max()
b9961_min = H24.filter(pl.col("build_num") == 9961)["ts"].min()
print("max ts for build 9842:", b9842_max, "| min ts for build 9961:", b9961_min)
print("build by game_directory (2024):")
print(H24.group_by(["build_num", "game_directory"]).len().sort(["build_num", "len"], descending=[False, True]).rows())

L24 = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet")
L24 = L24.join(H24.select(["key", "build_num", "ts", "map_name"]).rename({"key": "match"}), on="match", how="left")

print("\n== (b) 2024 metrics by build_num ==")
r13 = L24.filter((pl.col("round") == 13) & pl.col("scorable"))
print("round-13 score-vs-round_end by build (scorable rows):")
print(r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                    pl.col("sc_dis").mean().round(4).alias("rate")).sort("build_num"))
r13b = L24.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & pl.col("bomb").is_not_null())
print("round-13 e2 bomb mismatch by build (pc==20, bomb present):")
print(r13b.group_by("build_num").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm"),
                                     pl.col("bomb_mm").mean().round(4).alias("rate")).sort("build_num"))
post = L24.filter(pl.col("is_post") & pl.col("scorable"))
print("first-post-swap (D6 s17 definition) by build and s1 format:")
print(post.group_by(["build_num", "s1"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                             pl.col("sc_dis").mean().round(4).alias("rate")).sort(["build_num", "s1"]))
print("all scorable rounds by build and round (disagree>0 only, top 12):")
allsc = L24.filter(pl.col("scorable"))
print(allsc.group_by(["build_num", "round"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"))
      .filter(pl.col("dis") > 0).sort("dis", descending=True).head(12))
print("all scorable rounds by build (total):")
print(allsc.group_by("build_num").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis"),
                                      pl.col("sc_dis").mean().round(4).alias("rate")).sort("build_num"))
print("round-13 score-vs-round_end on build 9961 with md 2024-02-07 (exact rows, by timestamp):")
print(L24.filter((pl.col("round") == 13) & pl.col("scorable") & (pl.col("build_num") == 9961) & (pl.col("md") == "2024-02-07"))
      .group_by("sc_dis").len())

print("\n== (c) map mix by build (round-13 scorable rows) ==")
print(r13.group_by(["map_name", "build_num"]).agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort(["map_name", "build_num"]))
print("map counts by build (all 2024 matches):")
print(H24.group_by(["map_name", "build_num"]).len().sort(["map_name", "build_num"]))

print("\n== (d) 2026 header build_num and control ==")
H26 = hdr(BASE26, ["key", "match_date", "build_num", "game_directory", "network_protocol", "map_name"])
print("2026 build_num distribution:", H26.group_by("build_num").len().sort("build_num").rows())
print("2026 game_directory distribution:", H26.group_by("game_directory").len().sort("len", descending=True).head(6).rows())
L26 = pl.read_parquet(SCR / "verify_D6_6_long_2026.parquet")
L26 = L26.join(H26.select(["key", "build_num"]).rename({"key": "match"}), on="match", how="left")
allsc26 = L26.filter(pl.col("scorable"))
print("2026 all-round scorable by build:", allsc26.group_by("build_num").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("build_num").rows())
print("2026 first-post-swap scorable by build:",
      L26.filter(pl.col("is_post") & pl.col("scorable")).group_by("build_num").agg(pl.len().alias("n"), pl.col("sc_dis").sum().alias("dis")).sort("build_num").rows())

print("\n== (e) unscorable composition (2024 and 2026): why the score check is blank ==")
for lab, L in (("2024", L24), ("2026", L26)):
    tot = L.filter(pl.col("w_re").is_not_null())
    print(lab, "round_end rows:", tot.height, " with score winner:", tot.filter(pl.col("w_sc").is_not_null()).height,
          " without score winner:", tot.filter(pl.col("w_sc").is_null()).height)
