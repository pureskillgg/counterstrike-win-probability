"""Task C diagnostic 2: are round_state t_score/ct_score side-based or team-identity-based across swaps?

Hypotheses for the live-start scores of round r+1, given round r's live scores and round r's winner side:
  H_side : t_next = t + [winner == T];  ct_next = ct + [winner == CT]          (scores stay with the side)
  H_swap : t_next = ct + [winner == CT]; ct_next = t + [winner == T]          (scores follow the team across the swap)
A pair is 'swap boundary' when the round r+1 side assignment swaps (halftime after round 12, OT every 3 rounds).
Also looks at score_update deltas that are not +1.

Read-only, tomes only.
Run: pixi run python -I scripts\\eda\\priors_score_diag2.py
"""
import collections
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
    rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    su = read("score_update", ["match_key", "round", "team_code", "old_score", "new_score"])

    live = (rs.filter(pl.col("event_type") == "round_freeze_end")
            .sort(["match_key", "round", "tick"])
            .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
    d = (re_.join(live, on=["match_key", "round"], how="inner").join(hdr, on="match_key", how="left")
         .sort(["match_key", "round"]))
    d = d.with_columns([
        pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
        pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
        pl.col("round").shift(-1).over("match_key").alias("r_next"),
    ]).filter(pl.col("r_next") == pl.col("round") + 1)
    wT = (pl.col("winner_team_code") == 2)
    wC = (pl.col("winner_team_code") == 3)
    d = d.with_columns([
        ((pl.col("t_next") == pl.col("t_score") + wT.cast(pl.Int64)) & (pl.col("ct_next") == pl.col("ct_score") + wC.cast(pl.Int64))).alias("fit_side"),
        ((pl.col("t_next") == pl.col("ct_score") + wC.cast(pl.Int64)) & (pl.col("ct_next") == pl.col("t_score") + wT.cast(pl.Int64))).alias("fit_swap"),
    ]).with_columns(
        pl.when(pl.col("fit_side") & pl.col("fit_swap")).then(pl.lit("both"))
        .when(pl.col("fit_side")).then(pl.lit("side_only"))
        .when(pl.col("fit_swap")).then(pl.lit("swap_only"))
        .otherwise(pl.lit("neither")).alias("fit"),
        pl.when(pl.col("round") <= 24).then(pl.lit("reg")).otherwise(pl.lit("ot")).alias("bucket"),
    )
    print("pairs:", d.height)
    print("fit counts (all):", d.group_by("fit").len().sort("fit").rows())
    print("fit by bucket x wingman:")
    print(d.group_by(["bucket", "is_wingman", "fit"]).len().sort(["bucket", "is_wingman", "fit"]).rows())
    print("fit by round at boundary (reg, non-wingman, rounds 8..14):")
    print(d.filter((~pl.col("is_wingman")) & (pl.col("round").is_between(8, 14)))
          .group_by(["round", "fit"]).len().sort(["round", "fit"]).rows())
    print("fit for OT rounds 24..33 (non-wingman):")
    print(d.filter((~pl.col("is_wingman")) & (pl.col("round").is_between(24, 33)))
          .group_by(["round", "fit"]).len().sort(["round", "fit"]).rows())
    neither = d.filter(pl.col("fit") == "neither")
    print("neither rows by round (top 15):", neither.group_by("round").len().sort("len", descending=True).head(15).rows())
    print("neither rows by player_count:", neither.group_by("player_count").len().sort("player_count").rows())

    # score_update: non-unit deltas
    su2 = su.with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta"))
    odd = su2.filter(pl.col("delta").abs() != 1)
    print("\nscore_update rows:", su2.height, "non-unit deltas:", odd.height)
    print("non-unit delta example rows (first 8):", odd.head(8).rows())
    print("per (match, round) count of score_update rows (top):",
          su2.group_by(["match_key", "round"]).len()["len"].value_counts().sort("len").rows())


if __name__ == "__main__":
    main()
