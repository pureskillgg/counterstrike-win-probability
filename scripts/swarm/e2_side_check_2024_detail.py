"""Detail for the 2024 side-code mismatch found by e2_side_check.py.

For rounds with a defuse or explode row, flags rounds whose round_end winner disagrees with the bomb outcome
(defused -> winner should be 3 CT; exploded -> winner should be 2 T). Shows how the mismatches are spread over
matches and round numbers, and whether they look like whole-match side swaps or sporadic misalignment.
Run: pixi run python -I -X utf8 scripts\\swarm\\e2_side_check_2024_detail.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
BS = read("bomb_state", ["match_key", "round", "tick", "event_type"])
HDR = pl.concat([pl.read_parquet(f, columns=["key", "map_name"]) for f in
                 sorted(glob.glob(str(BASE / f"header" / "dataframe_*")))]).rename({"key": "match_key"})

ev = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
j = RE.filter(pl.col("player_count") == 20).join(ev, on=["match_key", "round"], how="inner").with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"),
).with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))

print(f"clean-20 rounds with bomb outcome: {j.height}; mismatches: {int(j['mismatch'].sum())}")
mm = j.filter(pl.col("mismatch"))
print(f"matches with >=1 mismatch: {mm['match_key'].n_unique()} of {j['match_key'].n_unique()} matches with bomb outcome")

per_match = (j.group_by("match_key").agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mm"))
             .with_columns((pl.col("mm") / pl.col("rounds")).alias("frac")))
print("\nper-match mismatch fraction distribution (rounds with bomb outcome):")
print(per_match.with_columns(
    pl.when(pl.col("frac") == 0).then(pl.lit("0")).when(pl.col("frac") < 0.1).then(pl.lit("<10%"))
    .when(pl.col("frac") < 0.5).then(pl.lit("10-50%")).when(pl.col("frac") < 1).then(pl.lit("50-99%"))
    .otherwise(pl.lit("100%")).alias("bucket")).group_by("bucket").len().sort("bucket"))

print("\nmismatch by round number (bucket of 6):")
print(mm.with_columns(((pl.col("round") - 1) // 6 * 6 + 1).alias("rbin")).group_by("rbin").len().sort("rbin"))

top = per_match.sort("mm", descending=True).head(15).join(HDR, on="match_key", how="left")
print("\ntop 15 matches by mismatch count:")
print(top)

# side-swap test per match: do mismatched rounds come in contiguous blocks (halftime-like)?
mm_rounds = mm.group_by("match_key").agg(pl.col("round").sort().alias("rounds"))
print("\nexample mismatched round lists (first 15 matches):")
for row in mm_rounds.head(15).iter_rows(named=True):
    print(row["match_key"][:12], row["rounds"])
