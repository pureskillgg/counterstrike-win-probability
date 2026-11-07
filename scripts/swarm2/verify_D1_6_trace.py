"""verify D1-6 trace (swarm2 verifier): per-round score trajectory for three 305-set matches and one control.

Prints, for rounds 1..11: round_end winner and reason, ADJ side from within-round change, the first and last
round_state rows (adjusted and raw scores), the number of round_state rows, and the round_end tick.
Purpose: find where the in-game score counters and the round_end winners diverge.
ASSUMPTIONS: A1 days from match_key; A8 duplicate round_end groups excluded (round 1 of 2f5bcb89...).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "winner_team_code", "player_count",
                                            "win_reason_message", "tick", "second"])
                for f in fl("round_end")], how="vertical_relaxed")
dups = RE.group_by(["match_key", "round"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
RE = RE.join(dups.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))

RS = pl.concat([pl.read_parquet(f, columns=["match_key", "round", "tick", "event_type",
                                            "t_score", "ct_score", "t_score_raw", "ct_score_raw", "phase"])
                for f in fl("round_state")], how="vertical_relaxed").sort(["match_key", "round", "tick"])
S = RS.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
    pl.col("t_score").last().alias("t1"), pl.col("ct_score").last().alias("c1"),
    pl.col("t_score_raw").first().alias("rt0"), pl.col("ct_score_raw").first().alias("rc0"),
    pl.col("t_score_raw").last().alias("rt1"), pl.col("ct_score_raw").last().alias("rc1"),
    pl.len().alias("n_states"), pl.col("tick").min().alias("tick_min"), pl.col("tick").max().alias("tick_max"),
    pl.col("phase").first().alias("phase0"))

M = RE.join(S, on=["match_key", "round"], how="left")
M = M.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
      .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("adj_side"))

set305 = (M.filter((pl.col("round") == 9) & (pl.col("day").is_in(DAYS_HI)) & (pl.col("player_count") == 20)
                   & (pl.col("adj_side").is_not_null()) & (pl.col("adj_side") != pl.col("winner_team_code")))
          ["match_key"].unique().sort().to_list())
print("305 set size:", len(set305))

picks = [set305[0], set305[1], set305[100]]
ctrl = (M.filter((pl.col("day").is_in(DAYS_HI)) & (pl.col("round") == 9) & (pl.col("player_count") == 20)
                 & (pl.col("adj_side") == pl.col("winner_team_code")))["match_key"].unique().sort().head(1).to_list())
for mk in picks + ctrl:
    tag = "305-SET" if mk in set305 else "CONTROL"
    hdr(f"{tag}: {mk}")
    x = M.filter((pl.col("match_key") == mk) & (pl.col("round") <= 13)).sort("round")
    print(x.select(["round", "winner_team_code", "win_reason_message", "adj_side", "player_count",
                    "t0", "c0", "t1", "c1", "rt0", "rc0", "rt1", "rc1", "n_states", "tick_min", "tick_max",
                    "tick", "phase0"]))
