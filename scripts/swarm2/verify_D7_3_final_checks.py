"""verify D7-3, final checks: (i) round-label offset in build 9842 (round 1 missing), (ii) flip drift by match date
(build 9961, score truth), (iii) which rounds the 'flip(s) elsewhere' matches flip on. Read-only; prints only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def chan_dirs(base, ch):
    return sorted(p for p in base.glob(f"basic_{ch}.*") if p.is_dir())


def read_chan(base, ch, cols, where=None):
    parts = []
    for d in chan_dirs(base, ch):
        for f in sorted(glob.glob(str(d / "dataframe_*"))):
            lf = pl.scan_parquet(f).select(cols)
            if where is not None:
                lf = lf.filter(where)
            parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed")


def read_header(base):
    fs = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name"]) for f in fs],
                     how="vertical_relaxed")


H = read_header(T24).rename({"key": "match_key"}).with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))

# (i) round labels in round_state by build
RS = read_chan(T24, "round_state", ["match_key", "round", "event_type", "phase"],
               where=pl.col("event_type") == "round_freeze_end").join(H.select(["match_key", "build_num"]), on="match_key")
print("=== (i) round_freeze_end labels by build ===")
print(RS.group_by("build_num").agg(pl.col("round").min().alias("min_round"), pl.col("round").max().alias("max_round"),
                                   pl.col("match_key").n_unique().alias("matches")).sort("build_num"))
fs = (RS.filter(pl.col("phase") == "second").group_by(["build_num", "match_key"]).agg(pl.col("round").min().alias("fs"))
      .group_by(["build_num", "fs"]).len().sort(["build_num", "len"], descending=[False, True]))
print("first 'second'-phase round, by build (top 6 per build):")
print(fs.group_by("build_num").head(6).sort(["build_num", "len"], descending=[False, True]))
ft = (RS.filter(pl.col("phase") == "first").group_by(["build_num", "match_key"]).agg(pl.col("round").max().alias("lf"))
      .group_by(["build_num", "lf"]).len().sort(["build_num", "len"], descending=[False, True]))
print("last 'first'-phase round, by build (top 6 per build):")
print(ft.group_by("build_num").head(6).sort(["build_num", "len"], descending=[False, True]))

# (ii) flip drift by match date, build 9961, round 13 (score truth)
S = pl.read_parquet(str(SCR / "verify_D7_3_score_rounds.parquet")).filter(pl.col("has_score"))
S = S.with_columns((pl.col("winner_team_code") != pl.col("winner_score")).alias("flip"))
m13 = S.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13)).with_columns(
    pl.col("mdt").dt.strftime("%m-%d").alias("mdd"))
d = m13.group_by("mdd").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"), pl.col("flip").mean().alias("rate")).sort("mdd")
print("\n=== (ii) build 9961 round-13 flip rate by match date (UTC) ===")
print(d)
print("first-half-of-period rate (mdd<=02-14):", round(m13.filter(pl.col("mdd") <= "02-14")["flip"].mean(), 4),
      "| second half (mdd>02-14):", round(m13.filter(pl.col("mdd") > "02-14")["flip"].mean(), 4))

# (iii) rounds flipped in build-9961 matches, by round number
f = S.filter((pl.col("build_num") == 9961) & pl.col("flip"))
print("\n=== (iii) flipped rounds, build 9961, by round number ===")
print(f.group_by("round").len().sort("round").rows())
mm = f.group_by("match_key").agg(pl.col("round").sort().alias("flip_rounds"))
print("distinct flipped-round sets per match (top 12):")
print(mm.group_by("flip_rounds").len().sort("len", descending=True).head(12))
