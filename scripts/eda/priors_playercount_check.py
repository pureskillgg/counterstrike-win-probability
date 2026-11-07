"""Task C check: is the player_count == 10 filter correlated with round number or outcome?

Read-only, tomes only. For non-wingman matches: player_count distribution by round bucket, and the CT win
rate of clean vs non-clean rounds by round bucket (to see whether the 5v5 filter selects on outcome).

Run: pixi run python -I scripts\\eda\\priors_playercount_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(ch, cols=None):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def main():
    hdr = read("header", ["key", "is_wingman"]).rename({"key": "match_key"})
    re_ = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
    d = re_.join(hdr, on="match_key", how="left").filter(~pl.col("is_wingman"))
    d = d.with_columns(pl.when(pl.col("round") > 24).then(pl.lit(25)).otherwise(pl.col("round")).alias("rb"),
                       (pl.col("player_count") == 10).alias("clean"),
                       (pl.col("winner_team_code") == 3).cast(pl.Int64).alias("ct"))
    print("non-wingman round_end rows:", d.height)
    tab = (d.group_by("rb").agg(pl.len().alias("rounds"), pl.col("clean").sum().alias("clean_rounds"),
                                pl.col("ct").mean().alias("ct_all"))
           .sort("rb").to_dicts())
    print("| round bucket | rounds | clean (10 players) | clean share | CT rate, clean | CT rate, not clean |")
    print("|---|---|---|---|---|---|")
    for r in tab:
        sub = d.filter(pl.col("rb") == r["rb"])
        c = sub.filter(pl.col("clean"))["ct"].mean()
        nc = sub.filter(~pl.col("clean"))["ct"].mean()
        nc_s = f"{nc:.3f}" if nc is not None else "n/a"
        print(f"| {r['rb']} | {r['rounds']} | {r['clean_rounds']} | {r['clean_rounds'] / r['rounds']:.3f} | "
              f"{c:.3f} | {nc_s} |")

    pc = d.group_by("player_count").len().sort("player_count").rows()
    print("\nplayer_count distribution (non-wingman):", pc)

    # matches with a clean round missing for some round 1..24
    clean_rounds = d.filter(pl.col("clean") & (pl.col("rb") <= 24))
    per_match = clean_rounds.group_by("match_key").agg(pl.col("round").n_unique().alias("n"))
    print("matches with clean rounds:", per_match.height,
          "with all 24 regulation rounds clean:", per_match.filter(pl.col("n") == 24).height)
    print("clean regulation rounds per match distribution:",
          per_match["n"].value_counts().sort("n").rows())


if __name__ == "__main__":
    main()
