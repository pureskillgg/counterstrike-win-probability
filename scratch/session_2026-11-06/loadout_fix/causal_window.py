"""Size the rows where the 22:46 causal bomb-lookup edit could change values. Read-only.

Old code: one plant position per round (the LAST plant event of the round) for every second of that round.
New code: the latest plant event at or before each tick (backward asof).
Values can differ only where a round has 2+ distinct plant ticks and the tick lies in [first plant, last plant).
"""
import polars as pl

FIN = "/home/sagemaker-user/adam/builder/final"
P = (pl.read_parquet(FIN + "/plants_all.parquet")
     .with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
     .unique(["match_key", "round", "tick"], keep="last", maintain_order=True))
print("plant events after (match, round, tick) dedupe:", P.height, flush=True)
g = (P.sort("tick").group_by(["match_key", "round"], maintain_order=True)
     .agg(pl.col("tick").min().alias("first_tick"), pl.col("tick").max().alias("last_tick"),
          pl.col("tick").n_unique().alias("n_ticks"),
          pl.col("x").first().alias("x_first"), pl.col("y").first().alias("y_first"),
          pl.col("x").last().alias("x_last"), pl.col("y").last().alias("y_last")))
multi = g.filter(pl.col("n_ticks") > 1)
diffpos = multi.filter((pl.col("x_first") != pl.col("x_last")) | (pl.col("y_first") != pl.col("y_last")))
print("rounds with a plant:", g.height, "| rounds with 2+ distinct plant ticks:", multi.height,
      "| of those, first and last plant positions differ:", diffpos.height, flush=True)
feat = ["c_min_bomb_dist", "t_min_bomb_dist", "c_n_bomb_500", "defuse_slack"]
tot = {f: 0 for f in feat}
for c in ["c050", "c060"]:
    d = pl.read_parquet(f"{FIN}/pos_parity/bucket_{c}.parquet", columns=["match_key", "round", "tick"] + feat)
    d = d.join(multi, on=["match_key", "round"], how="inner")
    win = d.filter((pl.col("tick") >= pl.col("first_tick")) & (pl.col("tick") < pl.col("last_tick")))
    win_diff = win.join(diffpos.select("match_key", "round"), on=["match_key", "round"], how="inner")
    print(f"bucket_{c}: parity rows in multi-plant rounds {d.height} | rows in [first, last) plant window {win.height} | "
          f"rows in window whose round has differing plant positions {win_diff.height}", flush=True)
    for f in feat:
        tot[f] += win_diff.filter(pl.col(f).is_not_null()).height
print("non-null bomb values in the window of differing-position rounds (c050+c060):", tot, flush=True)
