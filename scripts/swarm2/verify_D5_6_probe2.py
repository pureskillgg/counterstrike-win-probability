"""verify D5-6 probe 2: tick alignment of round_start, round_end, freeze_end and player_death within a few matches.

Reads the 2024 tome on G: with column projection. Prints one match in full and summary alignment stats.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_probe2.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(70)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs).select(cols)


RST = scan("round_start", ["match_key", "round", "tick", "second"]).collect()
RE = scan("round_end", ["match_key", "round", "tick", "second", "winner_team_code"]).collect()
DE = scan("player_death", ["match_key", "round", "tick", "second", "player_team_code"]).collect()

some = RST["match_key"].unique().sort()[:1].to_list()
mk = some[0]
print("example match_key:", mk)
print("round_start:")
print(RST.filter(pl.col("match_key") == mk).sort(["round", "tick"]).head(30))
print("round_end:")
print(RE.filter(pl.col("match_key") == mk).sort(["round", "tick"]).head(30))
print("player_death (first 25 by tick):")
print(DE.filter(pl.col("match_key") == mk).sort(["tick"]).head(25))

# global alignment: for each death, the round_start tick of the same label vs its round_end tick
rs_min = RST.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("st"))
re_min = RE.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("et"))
x = (DE.filter(pl.col("player_team_code").is_in([2, 3])).select(["match_key", "round", "tick"])
     .join(rs_min, on=["match_key", "round"], how="left").join(re_min, on=["match_key", "round"], how="left"))
print("\nvictim deaths:", x.height)
print("death tick - round_start tick (same label): quantiles")
print(x.select((pl.col("tick") - pl.col("st")).alias("d")).drop_nulls().select(
    pl.col("d").quantile(0.01).alias("q01"), pl.col("d").quantile(0.1).alias("q10"),
    pl.col("d").median().alias("med"), pl.col("d").quantile(0.9).alias("q90"), pl.col("d").quantile(0.99).alias("q99")))
print("fraction death tick < round_start tick (same label):",
      x.filter(pl.col("st").is_not_null()).select((pl.col("tick") < pl.col("st")).mean()).item())
print("fraction death tick > round_end tick (same label):",
      x.filter(pl.col("et").is_not_null()).select((pl.col("tick") > pl.col("et")).mean()).item())
print("tick ranges: round_start", RST["tick"].min(), RST["tick"].max(), " round_end", RE["tick"].min(), RE["tick"].max(),
      " death", DE["tick"].min(), DE["tick"].max())
