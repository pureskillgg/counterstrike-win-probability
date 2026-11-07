"""verify_D1_7 match-pattern and round-index checks (swarm2 verifier).

Q5 Are the 2024 non-swap disagreements (rounds 9, 13, 25, 28; d12-25) match-level (same matches disagree at several rounds)?
Q6 Round-index alignment test for round_end winner vs (corrected pre-swap scoreboard, bomb) at N+k, k in {-1, 0, +1},
   for 2024 pc20 d12-25 and 2026 pc10, to rule in or out a one-round mis-index of round_end rows.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(r"G:\pureskill\csds-cs2\scripts\swarm2")))
import verify_D1_7_recompute as R  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)

DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def q5(A):
    R.hdr("Q5 2024 pc20 d12-25: match-level pattern of winner vs corrected-score disagreements (rounds 1..30)")
    x = A.filter((pl.col("round") <= 30) & pl.col("day").is_in(DAYS_HI) & pl.col("corr_side").is_not_null())
    per = x.group_by("match_key").agg(
        pl.col("corr_dis").sum().alias("n_dis"),
        pl.col("round").filter(pl.col("corr_dis")).alias("dis_rounds"),
        pl.col("swap_sig").filter(pl.col("round") == 27).first().alias("swap27"))
    print("matches with any corrected disagreement:", per.filter(pl.col("n_dis") > 0).height, "of", per.height)
    print("disagreements per match (distribution):")
    print(per.group_by("n_dis").len().sort("n_dis").head(25))
    cnt = {}
    for r in per["dis_rounds"].to_list():
        for rr in r:
            cnt[rr] = cnt.get(rr, 0) + 1
    print("rounds carrying disagreements (count of matches), rounds 1..30:", dict(sorted(cnt.items())))
    from collections import Counter
    sets = Counter(",".join(str(v) for v in sorted(r)) for r in per.filter(pl.col("n_dis") > 0)["dis_rounds"].to_list())
    print("most common disagreement-round sets (round list per match: count):")
    for k_, v_ in sets.most_common(12):
        print(f"  [{k_}]: {v_}")


def q6(A, label):
    R.hdr(f"Q6 {label}: one-round index alignment of round_end winner vs other sources")
    base = A.filter(pl.col("round") <= 40).select(["match_key", "round", "winner_team_code", "corr_side", "bomb_side", "day"])
    for k in [-1, 0, 1]:
        other = base.select([pl.col("match_key"), (pl.col("round") - k).alias("round"),
                             pl.col("corr_side").alias("corr_k"), pl.col("bomb_side").alias("bomb_k")])
        # other row at round (N + k) lines up with winner at round N: other.round = N+k  -> use round shift
        j = base.join(other.with_columns((pl.col("round")).alias("round")), on=["match_key", "round"], how="inner")
        j = j.filter(pl.col("day").is_in(DAYS_HI))
        c = j.filter(pl.col("corr_k").is_not_null())
        b = j.filter(pl.col("bomb_k").is_not_null())
        print(f"k={k:+d}: winner vs corrected score at round+k agree {c.filter(pl.col('winner_team_code') == pl.col('corr_k')).height} of {c.height}"
              f"; winner vs bomb at round+k agree {b.filter(pl.col('winner_team_code') == pl.col('bomb_k')).height} of {b.height}")


def run():
    A24 = R.build(R.T24, "2024 tome", 20)
    q5(A24)
    q6(A24, "2024 pc20")
    A26 = R.build(R.T26, "2026 tome", 10)
    q6(A26, "2026 pc10")


if __name__ == "__main__":
    run()
