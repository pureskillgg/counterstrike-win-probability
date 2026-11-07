"""D1 cross-check (swarm2): score-side vs winner_team_code by round number in the 2026 tome.
Question: are the 2024 round-9/13/25/27/28 disagreements specific to 2024, or also present in 2026?
ASSUMPTIONS as in D1_flip_test_v2.py (A7 score side from round_state; player_count == 20 as in 2024 e2 check,
plus a second run with player_count == 10 as used by bakeoff.py).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")


def fl(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


def rd(ch, cols):
    files = fl(ch)
    print(f"{ch}: {len(files)} files")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
RS = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])
print("round_end rows:", RE.height, " round_state rows:", RS.height)
print("player_count distribution:", RE.group_by("player_count").len().sort("len", descending=True).head(8).to_dicts())

sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_before"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
        pl.col("t_score").sort_by("tick").last().alias("t_after"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
sc = sc.with_columns(
    pl.when((pl.col("t_after") - pl.col("t_before")) == 1).then(pl.lit(2))
      .when((pl.col("ct_after") - pl.col("ct_before")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))
J = RE.join(sc.select(["match_key", "round", "score_side"]), on=["match_key", "round"], how="inner")
J = J.filter(pl.col("score_side").is_not_null()).with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
print("\nrounds with a score side:", J.height)

for pc in [20, 10]:
    x = J.filter(pl.col("player_count") == pc)
    print(f"\n== player_count == {pc}: rounds with score side = {x.height}, disagreements = {int(x['dis'].sum())}")
    t = (x.group_by("round").agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"))
         .with_columns((pl.col("disagree") / pl.col("n")).round(4).alias("rate")).sort("round"))
    print(t.filter(pl.col("disagree") > 0))
    print("rounds 1..30 with zero disagreement:", t.filter(pl.col("round") <= 30).filter(pl.col("disagree") == 0).height,
          "of", t.filter(pl.col("round") <= 30).height)
    print("disagreement rate at round 13 / 25 / 28 / 27 / 9:")
    print(t.filter(pl.col("round").is_in([9, 13, 25, 27, 28])).sort("round"))
