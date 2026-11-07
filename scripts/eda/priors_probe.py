"""Task C probe: verify assumptions before the priors analysis (read-only, tomes only).

Checks: header key coverage, map and max_rounds mix, match_type/platform mix, round range and
overtime, live-start row per round, halftime phase boundary, and whether live-start scores equal
the cumulative score implied by the previous round's winner.

Run: pixi run python -I scripts\\eda\\priors_probe.py
"""
import collections
import glob
import os
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
    hdr = read("header")
    print("header columns:", hdr.columns)
    print("header rows:", hdr.height, "distinct key:", hdr["key"].n_unique())
    print("is_wingman:", hdr["is_wingman"].value_counts().sort("is_wingman").rows())
    print("map_name:", hdr["map_name"].value_counts().sort("count", descending=True).rows())
    print("max_rounds:", hdr["max_rounds"].value_counts().sort("max_rounds").rows())
    print("platform:", hdr["platform"].value_counts().sort("count", descending=True).rows())
    print("match_type:", hdr["match_type"].value_counts().sort("count", descending=True).rows())
    print("match_date day counts:",
          hdr.with_columns(pl.col("match_date").str.slice(0, 10).alias("d"))["d"].value_counts()
          .sort("d").rows())

    re_ = read("round_end")
    print("\nround_end rows:", re_.height, "matches:", re_["match_key"].n_unique())
    print("player_count:", re_["player_count"].value_counts().sort("player_count").rows())
    print("round max:", re_["round"].max(), "round min:", re_["round"].min())
    print("win reason:", re_["win_reason_code"].value_counts().sort("win_reason_code").rows())
    print("winner code:", re_["winner_team_code"].value_counts().sort("winner_team_code").rows())
    dup = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    print("duplicate (match_key, round) in round_end:", dup)

    # overtime: rounds beyond max_rounds
    joined = re_.join(hdr.select(["key", "max_rounds", "is_wingman"]).rename({"key": "match_key"}),
                      on="match_key", how="left")
    print("\nround_end rows with round > max_rounds:", joined.filter(pl.col("round") > pl.col("max_rounds")).height)
    print("round_end rows with null header join:", joined.filter(pl.col("max_rounds").is_null()).height)
    ot = joined.filter(pl.col("round") > 24)
    print("rounds > 24:", ot.height, "matches with rounds > 24:", ot["match_key"].n_unique())
    if ot.height:
        print("max round seen:", ot["round"].max())

    rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                              "t_score_raw", "ct_score_raw", "phase", "is_warmup"])
    print("\nround_state rows:", rs.height)
    print("phase values:", rs["phase"].value_counts().sort("phase").rows())
    fe = rs.filter(pl.col("event_type") == "round_freeze_end")
    print("round_freeze_end rows:", fe.height, "distinct (match, round):",
          fe.group_by(["match_key", "round"]).len().height)
    dup_fe = fe.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    print("(match, round) with >1 round_freeze_end:", dup_fe)
    rounds_with_fe = fe.select(["match_key", "round"]).unique()
    rounds_all = re_.select(["match_key", "round"]).unique()
    missing = rounds_all.join(rounds_with_fe, on=["match_key", "round"], how="anti").height
    print("round_end rounds without round_freeze_end:", missing)

    # halftime boundary: phase transitions by round, and round 12 vs 13
    ph = rs.group_by(["match_key", "round"]).agg(pl.col("phase").unique().alias("phases"))
    cnt = collections.Counter()
    for phases in ph["phases"].to_list():
        cnt[tuple(sorted(phases))] += 1
    print("phase sets per (match, round):", cnt.most_common(8))
    halftime_rounds = rs.filter(pl.col("phase") == "halftime").group_by("match_key").agg(
        pl.col("round").unique().sort().alias("rounds"))
    print("matches with halftime rows:", halftime_rounds.height)
    first_second = rs.filter(pl.col("phase") == "second").group_by(["match_key"]).agg(
        pl.col("round").min().alias("first_second_round"))
    print("first round in phase 'second' (count by value):",
          first_second["first_second_round"].value_counts().sort("first_second_round").rows()[:12])
    last_first = rs.filter(pl.col("phase") == "first").group_by(["match_key"]).agg(
        pl.col("round").max().alias("last_first_round"))
    print("last round in phase 'first' (count by value):",
          last_first["last_first_round"].value_counts().sort("last_first_round").rows()[:12])
    # wingman and 5v5 halves (non-wingman only, 24 rounds)
    wing = hdr.select(["key", "is_wingman"]).rename({"key": "match_key"})
    fs_w = first_second.join(wing, on="match_key").filter(~pl.col("is_wingman"))
    print("non-wingman first 'second' round counts:",
          fs_w["first_second_round"].value_counts().sort("first_second_round").rows()[:12])

    # live-start scores vs running totals from round_end winners (side based)
    live = fe.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").first(), pl.col("ct_score").first())
    wins = re_.select(["match_key", "round", "winner_team_code"])
    chk = live.join(wins, on=["match_key", "round"], how="inner").sort(["match_key", "round"])
    chk = chk.with_columns([
        pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
        pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
        pl.col("round").shift(-1).over("match_key").alias("r_next"),
    ])
    consecutive = chk.filter(pl.col("r_next") == pl.col("round") + 1)
    # live score for round r+1 should equal live score for round r plus the round r winner point
    t_inc = (pl.col("winner_team_code") == 2).cast(pl.Int64)
    c_inc = (pl.col("winner_team_code") == 3).cast(pl.Int64)
    consecutive = consecutive.with_columns([
        (pl.col("t_next") - pl.col("t_score") - t_inc).alias("t_err"),
        (pl.col("ct_next") - pl.col("ct_score") - c_inc).alias("ct_err"),
    ])
    print("\nlive-score consistency (r+1 live score - r live score - winner point):")
    print("  rows:", consecutive.height,
          "t_err nonzero:", consecutive.filter(pl.col("t_err") != 0).height,
          "ct_err nonzero:", consecutive.filter(pl.col("ct_err") != 0).height)
    print("live score at round 1 (t, ct):",
          live.filter(pl.col("round") == 1).select(["t_score", "ct_score"]).unique().rows()[:5])
    print("live rounds with null scores:", live.filter(pl.col("t_score").is_null()).height)

    # overtime sample: scores at the start of round 25 for matches that reached it
    r25 = live.filter(pl.col("round") >= 25).head(5)
    print("sample live rows round>=25:", r25.rows())


if __name__ == "__main__":
    main()
