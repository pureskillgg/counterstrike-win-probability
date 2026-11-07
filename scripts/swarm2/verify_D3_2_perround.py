"""D3-2 verifier: per-round disagreement table, days 12-25 vs days 01-07, 2024 G: tome (read-only).

For each round number r (player_count == 20, first round_end row per (match, round)):
  bomb rows (bomb outcome present):   n, round_end != bomb side, round_end != score chain (where defined)
  non-bomb rows (no bomb outcome):    n, round_end != score chain (where defined)
Score chain = cleaned freeze_end t/ct score change between r and r+1 (ASSUMPTION A2 in verify_D3_2_main.py).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def load(ch, cols):
    fs = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


RE = load("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
BS = load("bomb_state", ["match_key", "round", "event_type"])
RS = load("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
DAY = pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64)

re20 = (RE.filter(pl.col("player_count") == 20).sort("tick").group_by(["match_key", "round"])
        .agg(pl.col("winner_team_code").first().alias("re_w")))
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t_b"), pl.col("ct_score").first().alias("c_b")))
nxt = fe.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("t_b").alias("t_a"), pl.col("c_b").alias("c_a")])
sc = (fe.join(nxt, on=["match_key", "round"], how="left")
      .with_columns((pl.col("t_a") - pl.col("t_b")).alias("dt"), (pl.col("c_a") - pl.col("c_b")).alias("dc"))
      .with_columns(pl.when((pl.col("dc") == 1) & (pl.col("dt") == 0)).then(pl.lit(CT, dtype=pl.Int64))
                    .when((pl.col("dt") == 1) & (pl.col("dc") == 0)).then(pl.lit(T, dtype=pl.Int64))
                    .otherwise(pl.lit(None, dtype=pl.Int64)).alias("sc_win"))
      .select(["match_key", "round", "sc_win"]))
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
outc = outc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT, dtype=pl.Int64))
                         .otherwise(pl.lit(T, dtype=pl.Int64)).alias("bomb_side")).select(["match_key", "round", "bomb_side"])

A = (re20.join(sc, on=["match_key", "round"], how="left").join(outc, on=["match_key", "round"], how="left")
     .with_columns(DAY.alias("day"), pl.col("bomb_side").is_not_null().alias("has_bomb")))

for lab, cond in [("day01-07", pl.col("day") < 12), ("day12-25", pl.col("day") >= 12)]:
    S = A.filter(cond)
    bomb = S.filter(pl.col("has_bomb")).group_by("round").agg(
        pl.len().alias("bomb_n"),
        (pl.col("re_w") != pl.col("bomb_side")).sum().alias("re_ne_bomb"),
        (pl.col("sc_win").is_not_null() & (pl.col("re_w") != pl.col("sc_win"))).sum().alias("bomb_re_ne_score"))
    nb = S.filter(~pl.col("has_bomb")).group_by("round").agg(
        pl.len().alias("nobomb_n"),
        pl.col("sc_win").is_not_null().sum().alias("nobomb_sc_defined"),
        (pl.col("sc_win").is_not_null() & (pl.col("re_w") != pl.col("sc_win"))).sum().alias("nobomb_re_ne_score"))
    out = bomb.join(nb, on="round", how="full", coalesce=True).sort("round")
    print(f"\n== {lab}: per round number (bomb rows: RE vs bomb side and vs score; non-bomb: RE vs score)")
    print(out.filter((pl.col("bomb_n").fill_null(0) + pl.col("nobomb_n").fill_null(0)) > 0))

print("\n== DONE")
