"""verify D9-8, part 8: tick offset between the bomb outcome event and the round_end record, by build,
round bin and mismatch status (2024 tome, read-only). Offset = round_end tick - bomb outcome tick.
Mismatch = round_end winner_team_code != bomb side (e2 definition), player_count 20 only.
Output is printed; no files are written under E:.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


H = lz("header", ["key", "build_num"]).collect().rename({"key": "match_key"})
RE = lz("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE = RE.unique(subset=["match_key", "round"], keep="first")
BO = lz("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
OUT = BO.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    pl.col("tick").min().alias("bomb_tick"))
OUT = OUT.with_columns(pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"))
J = (RE.filter(pl.col("player_count") == 20).join(OUT, on=["match_key", "round"], how="inner")
       .join(H.select(["match_key", "build_num"]), on="match_key", how="left"))
J = J.with_columns(
    (pl.col("tick") - pl.col("bomb_tick")).alias("offset"),
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
      .otherwise(pl.lit("other")).alias("rbin"))
print("pc20 bomb-outcome rounds:", J.height)
print("\noffset = round_end tick - bomb outcome tick, by build x rbin x mismatch:")
print(J.group_by(["build_num", "rbin", "mismatch"]).agg(
    pl.len().alias("n"),
    (pl.col("offset") == 0).sum().alias("offset_zero"),
    (pl.col("offset") == 429).sum().alias("offset_429"),
    pl.col("offset").median().alias("median_offset"),
).sort(["build_num", "rbin", "mismatch"]))
print("\nbuild 9961 round 13 mismatch rows, offset value counts (top 10):")
print(J.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("mismatch"))
      .group_by("offset").len().sort("len", descending=True).head(10))
print("\nDONE")
