"""verify D5-6 debug: trace rounds whose build-table rw_pc is 8 against the raw round_end player_count.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_debug_pc.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(40)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_6")
K = ["match_key", "round"]
T = pl.read_parquet(SCR / "v_rounds.parquet")
print("T rows with rw_pc == 8:", T.filter(pl.col("rw_pc") == 8).height, " rw_pc value counts:")
print(T.group_by("rw_pc").agg(pl.len().alias("n")).sort("rw_pc"))
bad = T.filter(pl.col("rw_pc") == 8).select(K + ["re_tick", "rw"]).head(5)
print("sample pc==8 rows:", bad)
fs = sorted(glob.glob(str(BASE / f"basic_round_end{SUF}" / "dataframe_*")))
RE = pl.scan_parquet(fs).select(["match_key", "round", "tick", "player_count"]).collect()
for row in bad.iter_rows(named=True):
    sub = RE.filter((pl.col("match_key") == row["match_key"]) & (pl.col("round") == row["round"]))
    print("raw round_end rows for sample:", sub.height, sub.rows())
print("raw round_end player_count value counts (all rows):")
print(RE.group_by("player_count").agg(pl.len().alias("n")).sort("player_count"))
