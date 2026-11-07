r"""Verifier for D4-5, step A: 2024 header checks.

Question: the D4 claim assumes max_rounds = 24 for every 2024 match (swap_expected(r, 24)).
Does the 2024 header agree? Also compare the header final scores with the kept match count.

Reads only the header channel of the 2024 tome (columns projected). Writes nothing except stdout.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_5_a_header.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


H = read_dir(BASE / "header", ["key", "match_date", "t_starters_score_final",
                                "ct_starters_score_final", "match_type", "number_of_points", "tick_rate"])
print(f"2024 header rows: {H.height}; unique keys: {H['key'].n_unique()}")
print("(2024 header has NO max_rounds and NO freeze_time column; swap schedule is an ASSUMPTION)")
H = H.with_columns(
    pl.col("key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day_from_key"),
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("final_sum"),
)
print("\nkey sample:")
print(H.select(["key"]).head(3))
print("\nday_from_key distribution:")
print(H.group_by("day_from_key").len().sort("day_from_key"))
print("\nmatch_type counts:")
print(H.group_by("match_type").len().sort("len", descending=True))
print("\nnumber_of_points counts:")
print(H.group_by("number_of_points").len().sort("number_of_points"))
print("\ntick_rate counts:")
print(H.group_by("tick_rate").len().sort("tick_rate"))
print("\nfinal_sum distribution (starters' final scores; regulation 24 rounds max):")
print(H.group_by("final_sum").len().sort("final_sum"))
