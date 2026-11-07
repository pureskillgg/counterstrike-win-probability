"""verify_D1_7 three-source triad (swarm2 verifier).

For rows where both bomb evidence and a pre-swap scoreboard side exist, cross-tabulate
round_end winner_team_code vs bomb side vs corrected scoreboard side, for rounds 9, 13, 25, 27, 28.
Also reruns the round-index control for 2026 without the day filter (2026 days are 01-08).
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D1_7_recompute as R  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def triad(A, label):
    R.hdr(f"Triad {label}: winner vs bomb vs corrected scoreboard (rows with bomb AND corrected side)")
    x = A.filter(pl.col("bomb_side").is_not_null() & pl.col("corr_side").is_not_null())
    x = x.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
    for r in [9, 13, 25, 27, 28]:
        y = x.filter(pl.col("round") == r)
        if y.height == 0:
            print(f"round {r}: none")
            continue
        n = y.height
        w_b = y.filter(pl.col("winner_team_code") == pl.col("bomb_side")).height
        b_c = y.filter(pl.col("bomb_side") == pl.col("corr_side")).height
        w_c = y.filter(pl.col("winner_team_code") == pl.col("corr_side")).height
        print(f"round {r}: rows={n}  winner==bomb {w_b}  bomb==corr {b_c}  winner==corr {w_c}")
        print("   by day group (dg, winner==bomb, bomb==corr, winner==corr, rows):",
              y.group_by("dg").agg(
                  (pl.col("winner_team_code") == pl.col("bomb_side")).sum().alias("w_eq_b"),
                  (pl.col("bomb_side") == pl.col("corr_side")).sum().alias("b_eq_c"),
                  (pl.col("winner_team_code") == pl.col("corr_side")).sum().alias("w_eq_c"),
                  pl.len().alias("rows")).sort("dg").to_dicts())


def shift_control(A, label, all_days):
    R.hdr(f"Round-index control {label} (all days={all_days})")
    base = A.filter(pl.col("round") <= 40)
    if not all_days:
        base = base.filter(pl.col("day").is_in(DAYS_HI))
    base = base.select(["match_key", "round", "winner_team_code", "corr_side", "bomb_side"])
    S = A.filter(pl.col("round") <= 41).select(["match_key", "round", "corr_side", "bomb_side"])
    for k in [-1, 0, 1]:
        other = S.with_columns((pl.col("round") - k).alias("round")).rename({"corr_side": "corr_k", "bomb_side": "bomb_k"})
        j = base.join(other, on=["match_key", "round"], how="inner")
        c = j.filter(pl.col("corr_k").is_not_null())
        b = j.filter(pl.col("bomb_k").is_not_null())
        print(f"k={k:+d}: winner==corr(N+k) {c.filter(pl.col('winner_team_code') == pl.col('corr_k')).height} of {c.height}; "
              f"winner==bomb(N+k) {b.filter(pl.col('winner_team_code') == pl.col('bomb_k')).height} of {b.height}")


def run():
    A24 = R.build(R.T24, "2024 tome", 20)
    triad(A24, "2024 pc20")
    shift_control(A24, "2024 pc20", all_days=False)
    A26 = R.build(R.T26, "2026 tome", 10)
    triad(A26, "2026 pc10")
    shift_control(A26, "2026 pc10", all_days=True)


if __name__ == "__main__":
    run()
