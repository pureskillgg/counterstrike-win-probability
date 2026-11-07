"""verify D7-3: exact counts on phase-defined first-after-switch rounds (fs = first round with phase 'second'),
score truth, by build; and map pattern of the 9-round (halftime at 8) format. Read-only; prints only.
ASSUMPTION: round_state phase 'second' marks the first round after the halftime side switch.
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


H = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "map_name"])
               for f in sorted(glob.glob(str(T24 / "header" / "dataframe_*")))], how="vertical_relaxed") \
    .rename({"key": "match_key"})
RS = read_chan(T24, "round_state", ["match_key", "round", "event_type", "phase"],
               where=pl.col("event_type") == "round_freeze_end")
fs = (RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("fs")))
S = pl.read_parquet(str(SCR / "verify_D7_3_score_rounds.parquet")).filter(pl.col("has_score")) \
    .with_columns((pl.col("winner_team_code") != pl.col("winner_score")).alias("flip"))
S = S.join(fs, on="match_key", how="left").with_columns((pl.col("round") == pl.col("fs")).alias("is_fs"))

print("=== phase-defined first-after-switch rounds (score truth) by build ===")
print(S.filter(pl.col("is_fs")).group_by(["build_num", "round"]).agg(
    pl.len().alias("rows"), pl.col("flip").sum().alias("flips"), pl.col("flip").mean().alias("rate")).sort(["build_num", "round"]))
print("\nall other rounds by build (flips / rows):")
print(S.filter(~pl.col("is_fs")).group_by("build_num").agg(pl.col("flip").sum().alias("flips"), pl.len().alias("rows")))

# matches (build 9961) that have a fs round: did the fs round itself flip?
m = S.filter(pl.col("is_fs")).group_by(["match_key", "build_num"]).agg(pl.col("flip").any().alias("fs_flip"))
print("\nmatch-level, fs round flipped:", m.group_by(["build_num", "fs_flip"]).len().sort(["build_num", "fs_flip"]).rows())

# map pattern of the halftime-at-8 format (fs == 9)
M = fs.filter(pl.col("fs") == 9).join(H.select(["match_key", "build_num", "map_name"]), on="match_key", how="left")
print("\nmatches with first second-half round = 9 (halftime at 8), by build x map:")
print(M.group_by(["build_num", "map_name"]).len().sort(["build_num", "map_name"]).rows())
M13 = fs.filter(pl.col("fs") == 13).join(H.select(["match_key", "build_num", "map_name"]), on="match_key", how="left")
print("matches with first second-half round = 13 (halftime at 12), by build x map:")
print(M13.group_by(["build_num", "map_name"]).len().sort(["build_num", "map_name"]).rows())
