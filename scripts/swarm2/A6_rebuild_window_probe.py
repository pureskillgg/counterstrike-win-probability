"""A6 probe: is there a 2024 'rebuilt' round_end window?

notes/research/csds_docs_facts.md F6 (changelog 2024-02-12) says round_end rows built while the channel was
rebuilt from round_state carry only win codes 8 and 9, player_count 20 and legacy_code 0.
This probe tests that by match day (from match_key) and by round-end code family, on the raw 2024 round_end.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A6_rebuild_window_probe.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
fd = sorted(BASE.glob("basic_round_end.*"))[0]
RE = pl.scan_parquet(sorted(glob.glob(str(fd / "dataframe_*")))).select(
    ["match_key", "round", "winner_team_code", "win_reason_code", "player_count", "legacy_code"]).collect()
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/20\d\d/\d\d/(\d\d)/", 1).alias("day"),
                     pl.col("win_reason_code").is_in([8, 9]).alias("is89"),
                     (pl.col("player_count") == 20).alias("pc20"),
                     (pl.col("legacy_code") != 0).alias("legacy_nz"))
print("== by day: rows, share code 8/9, share player_count==20, share legacy!=0, count of codes outside 8/9")
out = RE.group_by("day").agg(
    pl.len().alias("rows"),
    pl.col("is89").mean().round(4).alias("share_89"),
    pl.col("pc20").mean().round(4).alias("share_pc20"),
    pl.col("legacy_nz").sum().alias("legacy_nz"),
    (~pl.col("is89")).sum().alias("codes_1_7_11_12_13_17_18")).sort("day")
print(out)
print("\n== codes 1/7/... by day (non 8/9 rows only)")
print(RE.filter(~pl.col("is89")).group_by(["day", "win_reason_code"]).len().sort(["day", "win_reason_code"]))
print("\n== player_count != 20 rows by day")
print(RE.filter(~pl.col("pc20")).group_by("day").len().sort("day"))
print("\n== days 01-07 vs 12-25: codes and pc20 for rows with legacy!=0")
print(RE.filter(pl.col("legacy_nz")).group_by(["day", "win_reason_code", "player_count"]).len().sort(["day", "win_reason_code"]))
