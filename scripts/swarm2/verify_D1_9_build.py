"""Build-number probe for claim D1-9 (verifier run). Read-only.

2024 header build_num is 9961 or 9842. Tests whether round_end.player_count depends on the build:
within each build, compare pc with the structure seen in player_info (nd, nT, nCT) and spawn (ns).
If build 9961 always gives pc = 20 regardless of structure (including 2v2 rounds), the 2024 pc is
a build-dependent constant and cannot be normalised by dividing by 2.
"""
import glob
import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(14)
pl.Config.set_fmt_str_lengths(64)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def files(ch):
    folder = TOME24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}")
    fs = sorted(glob.glob(str(folder / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no files for {ch}")
    return fs


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


hdr = read("header", ["key", "build_num"])
re_ = read("round_end", ["match_key", "round", "player_count"]).unique(["match_key", "round"], keep="first")
pi = read("player_info", ["match_key", "round", "team_code", "player_id_fixed"])
pr = pi.group_by(["match_key", "round"]).agg(
    pl.col("player_id_fixed").n_unique().alias("nd"),
    pl.col("player_id_fixed").filter(pl.col("team_code") == T).n_unique().alias("nT"),
    pl.col("player_id_fixed").filter(pl.col("team_code") == CT).n_unique().alias("nCT"))
sp = read("player_spawn", ["match_key", "round", "player_id_fixed"])
sr = sp.group_by(["match_key", "round"]).agg(pl.col("player_id_fixed").n_unique().alias("ns"))
R = (re_.join(pr, on=["match_key", "round"], how="left")
        .join(sr, on=["match_key", "round"], how="left")
        .join(hdr.rename({"key": "match_key"}), on="match_key", how="left"))

print("== rounds by build_num: player_count value (top 10) ==")
print(R.group_by(["build_num", "player_count"]).len().sort(["build_num", "len"], descending=[False, True])
      .group_by("build_num").head(10).sort(["build_num", "len"], descending=[False, True]))

print("\n== structure (player_info nT x nCT, rounds with player_info) by build_num, top 8 each ==")
for b in sorted(R["build_num"].drop_nulls().unique().to_list()):
    sub = R.filter(pl.col("build_num") == b).filter(pl.col("nT").is_not_null())
    print(f"\nbuild {b}: rounds with player_info {sub.height:,}")
    print(sub.group_by(["nT", "nCT"]).len().sort("len", descending=True).head(8))

print("\n== build x structure x player_count: 2v2 (nT=2,nCT=2) and 5v5 (nT=5,nCT=5) ==")
kind = R.with_columns(
    pl.when((pl.col("nT") == 2) & (pl.col("nCT") == 2)).then(pl.lit("2v2"))
    .when((pl.col("nT") == 5) & (pl.col("nCT") == 5)).then(pl.lit("5v5"))
    .otherwise(pl.lit("other/null")).alias("kind"))
print(kind.group_by(["build_num", "kind", "player_count"]).len().sort(["build_num", "kind", "len"],
                                                                     descending=[False, False, True]))

print("\n== build x spawn-based kind (ns) x player_count, top 20 ==")
print(R.filter(pl.col("ns").is_not_null()).group_by(["build_num", "ns", "player_count"]).len()
      .sort("len", descending=True).head(20))

print("\n== match level: build x modal structure ==")
mm = kind.group_by(["build_num", "match_key"]).agg(
    pl.col("kind").mode().first().alias("kind_mode"),
    pl.col("player_count").mode().first().alias("pc_mode"))
print(mm.group_by(["build_num", "kind_mode", "pc_mode"]).len().sort(["build_num", "len"], descending=[False, True]).head(30))
