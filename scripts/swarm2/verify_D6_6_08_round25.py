"""D6-6 verifier step 08: round 25 (first overtime round in 24-round matches) by build, bomb-based only (score rule
cannot score round 25), and whether the 6 stray build-9961 round-9 disagreements sit in matches with no
player_info swap (D6 section 21 undetermined matches).
Read-only.
Run: "G:/pureskill/csds-cs2/.pixi/envs/default/python.exe" -I -X utf8 scripts/swarm2/verify_D6_6_08_round25.py
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
SCR = Path(os.environ.get("VERIFY_SCRATCH", r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"))
BASE24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"

H = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "match_date", "map_name"])
               for f in sorted(glob.glob(str(BASE24 / "header" / "dataframe_*")))], how="vertical_relaxed")
H = H.unique(["key"], keep="first", maintain_order=True).rename({"key": "match"})
L = pl.read_parquet(SCR / "verify_D6_6_long_2024.parquet").join(H.select(["match", "build_num", "map_name"]), on="match", how="left")

print("== round 25: rows with round_end, score winner, bomb outcome; by build ==")
r25 = L.filter(pl.col("round") == 25)
print(r25.group_by("build_num").agg(
    pl.len().alias("round_end_rows"),
    pl.col("w_sc").is_not_null().sum().alias("score_winner_present"),
    (pl.col("bomb").is_not_null() & pl.col("w_re").is_not_null()).sum().alias("bomb_rows"),
    pl.col("bomb_mm").sum().alias("bomb_mismatch")).sort("build_num"))
b25 = L.filter((pl.col("round") == 25) & pl.col("bomb").is_not_null() & pl.col("w_re").is_not_null())
print("round 25 bomb rows by build and player_count:")
print(b25.group_by(["build_num", "pc"]).agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm")).sort(["build_num", "pc"]))

print("\n== round 25 bomb mismatch by map, build 9961 ==")
print(b25.filter(pl.col("build_num") == 9961).group_by("map_name").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm")).sort("map_name"))

print("\n== the 6 stray round-9 disagreements (build 9961, no player_info swap): score-derived swap at 8? ==")
six_keys = ["csds/2024/02/15/6bb16fb7-9861-480b-ba0d-6ad26dce10ae/csds", "csds/2024/02/20/14ac89df-f38a-43fb-9cfc-4d8d511046db/csds",
            "csds/2024/02/20/a052e290-776b-42b0-8c7c-15570bf03fdf/csds", "csds/2024/02/20/af1c76fb-e57b-4f64-89ac-010a65feb63e/csds",
            "csds/2024/02/21/f57e6986-6209-40e5-a0bf-79dd3f0952a8/csds", "csds/2024/02/22/132c429e-f5d1-470e-ae05-58e9e3b4807d/csds"]
for k in six_keys:
    rows = L.filter((pl.col("match") == k) & (pl.col("round").is_in([8, 9, 10]))).select(
        ["round", "w_sc", "w_re", "bomb", "scorable", "sc_dis"]).sort("round").rows()
    print(k[:44], rows)
print("\nundetermined (no player_info swap) matches in 2024, by build:")
und = L.filter(pl.col("s1").is_null()).select("match", "build_num").unique()
print(und.group_by("build_num").len().sort("build_num").rows())
