# N1_lopsided check: compare death windows (round_end tick vs grid's own last row).
import glob
import numpy as np
import polars as pl

TOME = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
D = "G:/pureskill/csds-cs2/data"
K = ["match_key", "round"]


def pages(ch):
    return sorted(glob.glob(TOME + "/basic_" + ch + ".COMBINED/dataframe_*"))


grid = pl.read_parquet(D + "/viz/per_tick_states.parquet", columns=["match_key", "round", "tick", "seconds_since_round_live"])
grid_matches = sorted(grid["match_key"].unique().to_list())
live = grid.filter(pl.col("seconds_since_round_live") == 0).group_by(K).agg(pl.col("tick").min().alias("live_start_tick"))
last = grid.group_by(K).agg(pl.col("tick").max().alias("grid_last_tick"))
re_all = pl.concat([pl.read_parquet(p) for p in pages("round_end")], how="diagonal_relaxed")
re_u = (re_all.filter(pl.col("match_key").is_in(grid_matches))
        .select(K + ["tick", "winner_team_code"]).unique(subset=K, keep="first")
        .rename({"tick": "round_end_tick"}))
rounds = live.join(re_u, on=K, how="inner").join(last, on=K, how="inner")
d = (rounds["round_end_tick"] - rounds["grid_last_tick"]).to_numpy()
print("round_end_tick - grid_last_tick: min", d.min(), "median", np.median(d), "max", d.max(),
      "frac grid_last >= round_end", float((d <= 0).mean()))
pdeath = pl.concat([pl.read_parquet(p, columns=["match_key", "round", "tick", "player_id", "player_team_code"])
                    for p in pages("player_death")], how="diagonal_relaxed")
pdeath = pdeath.filter(pl.col("match_key").is_in(grid_matches)).unique(subset=K + ["tick", "player_id"], keep="first")
pdeath = pdeath.filter(pl.col("player_team_code").is_in([2, 3]))
j = pdeath.join(rounds, on=K, how="inner")
tick = j["tick"].to_numpy()
ls = j["live_start_tick"].to_numpy()
re_t = j["round_end_tick"].to_numpy()
gl = j["grid_last_tick"].to_numpy()
w1 = (tick >= ls) & (tick <= re_t)
w2 = (tick >= ls) & (tick <= gl)
w3 = (tick >= ls) & (tick <= np.maximum(re_t, gl))
print("deaths (team known, grid rounds):", len(tick))
print("in [live, round_end]:", int(w1.sum()), " in [live, grid_last]:", int(w2.sum()), " in [live, max(end,last)]:", int(w3.sum()))
after = (tick > re_t) & (tick >= ls)
print("deaths after round_end, by whether also <= grid_last:", int((after & (tick <= gl)).sum()), "of", int(after.sum()))
# per-round death totals under windows, and winner/loser split for window max(end,last)
ids = j.select(K + ["player_team_code"]).with_columns(pl.Series("w1", w1), pl.Series("w3", w3))
w = rounds.select(K + ["winner_team_code"])
for col in ["w1", "w3"]:
    per = ids.filter(pl.col(col)).group_by(K).agg([
        (pl.col("player_team_code") == 2).sum().alias("dt"),
        (pl.col("player_team_code") == 3).sum().alias("dct")])
    r = rounds.select(K + ["winner_team_code"]).join(per, on=K, how="left").with_columns([
        pl.col("dt").fill_null(0), pl.col("dct").fill_null(0)])
    win = np.where(r["winner_team_code"].to_numpy() == 3, r["dct"].to_numpy(), r["dt"].to_numpy())
    lose = np.where(r["winner_team_code"].to_numpy() == 3, r["dt"].to_numpy(), r["dct"].to_numpy())
    print(col, "winners mean", round(float(win.mean()), 4), "losers mean", round(float(lose.mean()), 4),
          "winner hist", np.bincount(win.astype(np.int64), minlength=6)[:6].tolist(),
          "loser hist", np.bincount(lose.astype(np.int64), minlength=6)[:6].tolist())
