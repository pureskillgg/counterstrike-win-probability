"""A3: round-13 bomb-outcome mismatch (e2 definition) grouped by the header match_date (full date)
and by the key-folder date used in scripts/swarm/e2_side_check_r13.py. Read-only.

Definition as in e2_side_check_r13.py: round_end.player_count == 20 joined to bomb_defused or
bomb_exploded; bomb_side 3 if defused else 2; mismatch = bomb_side != winner_team_code.
Duplicate copies are kept here (this is the e2 population); the copy-removed view is in
A3_dedupe_2024_impact.py.
ASSUMPTION: match_date is not verified as the played date (bakeoff_results.md says so for 2026).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def read(ch, cols):
    fd = BASE / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def bucket(d):
    if d < "2024-01-01":
        return "2023-12"
    if d < "2024-02-01":
        return "2024-01"
    if d <= "2024-02-07":
        return "2024-02-01..07"
    if d <= "2024-02-11":
        return "2024-02-08..11"
    if d <= "2024-02-25":
        return "2024-02-12..25"
    return "after 2024-02-25"


def main():
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
    BS = read("bomb_state", ["match_key", "round", "event_type"])
    H = read("header", ["key", "match_date"]).rename({"key": "match_key"})
    H = H.with_columns(pl.col("match_date").str.slice(0, 10).alias("md_date"))
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    j = (RE.filter(pl.col("player_count") == 20)
         .join(outcome, on=["match_key", "round"], how="inner")
         .join(H, on="match_key", how="left")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
             pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("key_day"))
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                       pl.col("md_date").map_elements(bucket, return_dtype=pl.Utf8).alias("md_bucket"),
                       pl.format("2024-02-{}", pl.col("key_day")).alias("key_date")))
    r13 = j.filter(pl.col("round") == 13)
    print(f"round 13 rows with outcome: {r13.height}; mismatches {int(r13['mismatch'].sum())}")

    print("\nround 13 by key-folder date (e2 view):")
    g = (r13.group_by("key_date").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
         .sort("key_date"))
    print(g)

    print("\nround 13 by match_date bucket (full calendar date):")
    g2 = (r13.group_by("md_bucket").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
          .with_columns((pl.col("mm") / pl.col("n")).round(3).alias("rate")).sort("md_bucket"))
    print(g2)

    print("\nround 13 by match_date, Feb 2024 day (full date):")
    g3 = (r13.filter(pl.col("md_date").str.starts_with("2024-02"))
          .group_by("md_date").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
          .sort("md_date"))
    print(g3)

    same = r13.filter(pl.col("key_date") == pl.col("md_date")).height
    print(f"\nround-13 rows where key-folder date equals match_date: {same} of {r13.height}")
    all_rows_mm = int(j["mismatch"].sum())
    print(f"all rounds with outcome (any round): rows {j.height}; mismatches {all_rows_mm}")


if __name__ == "__main__":
    main()
