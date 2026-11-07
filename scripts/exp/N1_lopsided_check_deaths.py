# N1_lopsided check: where do the null-team deaths and out-of-window deaths sit?
import glob
import polars as pl

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
D = "G:/pureskill/csds-cs2/data"
K = ["match_key", "round"]


def pages(ch):
    return sorted(glob.glob(TOME + "/basic_" + ch + ".COMBINED/dataframe_*"))


grid = pl.read_parquet(D + "/viz/per_tick_states.parquet", columns=["match_key", "round", "tick", "seconds_since_round_live"])
grid_matches = sorted(grid["match_key"].unique().to_list())
live = grid.filter(pl.col("seconds_since_round_live") == 0).group_by(K).agg(pl.col("tick").min().alias("live_start_tick"))
last = grid.group_by(K).agg(pl.col("tick").max().alias("grid_last_tick"), pl.col("tick").min().alias("grid_first_tick"))
re_all = pl.concat([pl.read_parquet(p) for p in pages("round_end")], how="diagonal_relaxed")
re_u = (re_all.filter(pl.col("match_key").is_in(grid_matches))
        .select(K + ["tick"]).unique(subset=K, keep="first").rename({"tick": "round_end_tick"}))
rounds = live.join(re_u, on=K, how="inner").join(last, on=K, how="inner")
pdeath = pl.concat([pl.read_parquet(p, columns=["match_key", "round", "tick", "player_id", "player_team_code"])
                    for p in pages("player_death")], how="diagonal_relaxed")
pdeath = pdeath.filter(pl.col("match_key").is_in(grid_matches)).unique(subset=K + ["tick", "player_id"], keep="first")
j = pdeath.join(rounds, on=K, how="inner")
j = j.with_columns(
    pl.when(pl.col("tick") < pl.col("live_start_tick")).then(pl.lit("before_live"))
    .when(pl.col("tick") > pl.col("round_end_tick")).then(pl.lit("after_round_end"))
    .otherwise(pl.lit("inside")).alias("where"),
    (pl.col("tick") - pl.col("round_end_tick")).alias("ticks_after_end"),
    (pl.col("tick") - pl.col("live_start_tick")).alias("ticks_after_live"),
)
print("deaths joined to grid rounds:", j.height)
print(j.group_by(["where", pl.col("player_team_code").fill_null(-1)]).len().sort(["where", "player_team_code"]).rows())
after = j.filter(pl.col("where") == "after_round_end")
print("after_round_end ticks_after_end describe:")
print(after["ticks_after_end"].describe().rows())
print("after_round_end: count within 64 ticks (1 s):", after.filter(pl.col("ticks_after_end") <= 64).height,
      "within 256 ticks:", after.filter(pl.col("ticks_after_end") <= 256).height)
before = j.filter(pl.col("where") == "before_live")
print("before_live ticks_after_live describe:")
print(before["ticks_after_live"].describe().rows())
nulls = j.filter(pl.col("player_team_code").is_null())
print("null-team deaths by where:", nulls.group_by("where").len().rows())
print("null-team deaths in grid rounds total", nulls.height)
