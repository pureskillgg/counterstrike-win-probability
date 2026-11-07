"""Task C diagnostic: why live-start scores do not always step by the previous round's winner point.

Read-only, tomes only. Groups the r -> r+1 live-score mismatches by win reason, winner side, round
range and wingman flag, and checks whether the score change comes from score_update instead.

Run: pixi run python -I scripts\\eda\\priors_score_diag.py
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
    hdr = read("header", ["key", "is_wingman", "map_name", "max_rounds"]).rename({"key": "match_key"})
    re_ = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    su = read("score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])

    live = (rs.filter(pl.col("event_type") == "round_freeze_end")
            .sort(["match_key", "round", "tick"])
            .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first(),
                                                   pl.col("tick").first().alias("live_tick")))
    d = (re_.join(live, on=["match_key", "round"], how="left")
         .join(hdr, on="match_key", how="left")
         .sort(["match_key", "round"]))
    d = d.with_columns([
        pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
        pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
        pl.col("round").shift(-1).over("match_key").alias("r_next"),
        pl.col("live_tick").shift(-1).over("match_key").alias("live_next"),
    ])
    d = d.filter(pl.col("r_next") == pl.col("round") + 1)
    d = d.with_columns([
        (pl.col("t_next") - pl.col("t_score") - (pl.col("winner_team_code") == 2).cast(pl.Int64)).alias("t_err"),
        (pl.col("ct_next") - pl.col("ct_score") - (pl.col("winner_team_code") == 3).cast(pl.Int64)).alias("ct_err"),
    ])
    bad = d.filter((pl.col("t_err") != 0) | (pl.col("ct_err") != 0))
    print("pairs:", d.height, "mismatch:", bad.height)
    print("\nmismatch by win_reason_code:")
    print(bad.group_by("win_reason_code").len().sort("win_reason_code").rows())
    print("\nmismatch by is_wingman:", bad.group_by("is_wingman").len().rows())
    print("\nmismatch by round bucket:")
    print(bad.with_columns(pl.when(pl.col("round") <= 24).then(pl.lit("reg")).otherwise(pl.lit("ot")).alias("b"))
          .group_by("b").len().rows())
    print("\nmismatch by player_count:", bad.group_by("player_count").len().sort("player_count").rows())
    print("\nmismatch by winner:", bad.group_by("winner_team_code").len().rows())
    print("\nmismatch by (round_next - round) (should be 1):", bad.with_columns((pl.col("r_next") - pl.col("round")).alias("gap")).group_by("gap").len().rows())

    print("\nfirst 12 mismatch rows (key suffix, round, winner, reason, t_score, t_next, ct_score, ct_next, t_err, ct_err):")
    for row in bad.head(12).iter_rows(named=True):
        print(row["match_key"][-40:], row["round"], row["winner_team_code"], row["win_reason_code"],
              row["t_score"], row["t_next"], row["ct_score"], row["ct_next"], row["t_err"], row["ct_err"])

    # Score updates per match and round: does score_update agree with round_end winners?
    su_chk = (su.join(re_.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="left")
              .with_columns((pl.col("new_score") - pl.col("old_score")).alias("delta")))
    print("\nscore_update rows:", su.height, " delta distribution:", su_chk["delta"].value_counts().sort("delta").rows())
    print("score_update rows whose team_code != winner_team_code:",
          su_chk.filter(pl.col("team_code") != pl.col("winner_team_code")).height)

    # Does round_state t_score at the round_end event carry the NEW score?
    re_state = rs.filter(pl.col("event_type") == "round_end").select(["match_key", "round", "t_score", "ct_score"])
    print("round_state round_end rows:", re_state.height)
    cmp = re_state.join(live.select(["match_key", "round", "t_score", "ct_score"]), on=["match_key", "round"],
                        suffixes=("_end", "_live"))
    print("round_end state score minus live-start score (T, CT) distribution:",
          cmp.with_columns((pl.col("t_score_end") - pl.col("t_score_live")).alias("dt"),
                           (pl.col("ct_score_end") - pl.col("ct_score_live")).alias("dct"))
          .group_by(["dt", "dct"]).len().sort(["dt", "dct"]).rows())


if __name__ == "__main__":
    main()
