import os
import numpy as np
import polars as pl

BASE = r"G:\pureskill\csds-cs2\data"
PLAYERS = os.path.join(BASE, "state_sample", "players_1hz.parquet")
ROUNDS = os.path.join(BASE, "state_sample", "rounds.parquet")

rounds = pl.read_parquet(ROUNDS).with_columns(pl.col("round").cast(pl.Int64))
print("raw all-round CT win rate (2673 rows):", round(float(rounds["ct_win"].mean()), 6),
      " count ct_win=1:", int(rounds["ct_win"].sum()))

r2 = rounds.sort(["match_key", "round"]).with_columns(
    pl.col("ct_score").shift(-1).over("match_key").alias("ct_next"),
    pl.col("t_score").shift(-1).over("match_key").alias("t_next"),
    pl.col("round").shift(-1).over("match_key").alias("round_next"),
)
pairs = r2.filter(pl.col("ct_next").is_not_null())
pairs = pairs.with_columns(
    (pl.col("ct_next") - pl.col("ct_score")).alias("d_ct"),
    (pl.col("t_next") - pl.col("t_score")).alias("d_t"),
)
print("pairs:", pairs.height)
print("round_next - round == 1 for all pairs:", bool((pairs["round_next"] - pairs["round"] == 1).all()))
print("d_ct + d_t distribution:", pairs.group_by((pl.col("d_ct") + pl.col("d_t"))).agg(pl.len().alias("n")).sort("n", descending=True).to_dicts())
print("d_ct distribution by ct_win:")
print(pairs.group_by(["ct_win", "d_ct"]).agg(pl.len().alias("n")).sort(["ct_win", "d_ct"]).to_dicts())

# mismatch under the stated convention
mis = pairs.filter(~((pl.col("d_ct") == pl.col("ct_win")) & (pl.col("d_t") == 1 - pl.col("ct_win"))))
print("mismatch count:", mis.height)
print("mismatch by round (first 24 rounds):")
print(mis.group_by("round").agg(pl.len().alias("n")).sort("round").to_dicts())
print("mismatch by round half: round<=12:", mis.filter(pl.col("round") <= 12).height,
      " round>=13:", mis.filter(pl.col("round") >= 13).height)
print("pairs by round half: round<=12:", pairs.filter(pl.col("round") <= 12).height,
      " round>=13:", pairs.filter(pl.col("round") >= 13).height)
print("mismatch sample rows:")
print(mis.select(["match_key", "round", "ct_win", "ct_score", "t_score", "ct_next", "t_next", "d_ct", "d_t", "map_name", "split"]).head(12))
print("mismatch matches:", mis["match_key"].n_unique(), "of", rounds["match_key"].n_unique())

# does the ct_score/t_score increment ever disagree with ct_win when the round is pistol-ish or the match is short?
print("rounds with plant_tick null:", rounds.filter(pl.col("plant_tick").is_null()).height)
print("match-level mismatch counts (top 10):")
print(mis.group_by("match_key").agg(pl.len().alias("n")).sort("n", descending=True).head(10).to_dicts())

# check round_number vs round and final scores per match
fin = rounds.sort(["match_key", "round"]).group_by("match_key").agg([
    pl.col("ct_score").last().alias("ct_last_pre"),
    pl.col("t_score").last().alias("t_last_pre"),
    pl.col("ct_win").last().alias("last_ct_win"),
    pl.len().alias("n_rounds"),
    pl.col("ct_win").sum().alias("ct_wins_total"),
])
print("match n_rounds distribution:", fin.group_by("n_rounds").agg(pl.len().alias("matches")).sort("n_rounds").to_dicts())
fin2 = fin.with_columns(
    (pl.col("ct_wins_total") == (pl.col("ct_last_pre") + pl.col("last_ct_win"))).alias("ct_total_consistent"),
    ((pl.col("n_rounds") - pl.col("ct_wins_total")) == (pl.col("t_last_pre") + (1 - pl.col("last_ct_win")))).alias("t_total_consistent"),
)
print("match-level totals: ct_wins_total == final ct_score + last ct_win :", int(fin2["ct_total_consistent"].sum()), "of", fin2.height)
print("match-level totals: t wins total == final t_score + (1-last ct_win):", int(fin2["t_total_consistent"].sum()), "of", fin2.height)

# which round was excluded by the 10-player filter
p0 = (pl.scan_parquet(PLAYERS)
      .filter(pl.col("seconds_since_live") == 0)
      .select(["match_key", "round", "tick", "player_id", "team_code", "equipment_value_calc"])
      .collect()
      .with_columns(pl.col("round").cast(pl.Int64)))
cnt = p0.group_by(["match_key", "round"]).agg([
    pl.len().alias("n_rows"),
    pl.col("player_id").n_unique().alias("n_pid"),
    (pl.col("team_code") == 3).sum().alias("n_ct"),
    (pl.col("team_code") == 2).sum().alias("n_t"),
    pl.col("tick").n_unique().alias("n_tick"),
])
bad = cnt.filter((pl.col("n_rows") != 10) | (pl.col("n_pid") != 10))
print("rounds with non-10 player rows at sec0:", bad.to_dicts())
allkeys = rounds.select(["match_key", "round"])
missing = allkeys.join(cnt.select(["match_key", "round"]), on=["match_key", "round"], how="anti")
print("rounds with no sec0 rows:", missing.height)
