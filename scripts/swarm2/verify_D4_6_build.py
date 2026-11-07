"""Build a per-round table for the D4-6 verifier (2024 tome, full rounds and all rounds).

One row per (match_key, round). Sources (all read with column projection, one channel at a time):
  round_state : first round_freeze_end (live start) and its t_score/ct_score, plus the next round's freeze_end scores
  round_end   : first round_end row (end tick, winner_team_code, win reason, player_count)
  player_death: deaths with player_team_code in {2,3}; last death tick inside [live, end]
  bomb_state  : planted tick, defused/exploded tick
ASSUMPTIONS
  A1. Tick rate 64 per second (so 192 ticks = 3 s, 429 ticks = 6.7 s).
  A2. Score side S: CT if CT score rose by 1 and T score did not, T if T rose by 1 and CT did not, else 0 (same rule as D4).
  A3. Last death = max tick of deaths with live <= tick <= end; victim sides at that tick must be one side (ties dropped).
Output: scratchpad parquet, not in the project tree.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
ROOT = Path(r"G:\pureskill\csds-cs2")
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
SCRATCH.mkdir(parents=True, exist_ok=True)
SUF = ".2023-12-28,2024-02-25"
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3


def read(ch, cols):
    fd = BASE / f"basic_{ch}{SUF}"
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
fe = (rs.filter(pl.col("event_type") == "round_freeze_end")
      .sort(["match_key", "round", "tick"])
      .unique(["match_key", "round"], keep="first")
      .select(["match_key", "round", pl.col("tick").alias("live"), "t_score", "ct_score", "t_score_raw", "ct_score_raw"]))
nxt = (fe.with_columns((pl.col("round") - 1).alias("round"))
       .select(["match_key", "round", pl.col("t_score").alias("t_next"), pl.col("ct_score").alias("ct_next"),
                pl.col("t_score_raw").alias("t_next_raw"), pl.col("ct_score_raw").alias("ct_next_raw")]))
R = fe.join(nxt, on=["match_key", "round"], how="left")


def side_expr(t0, t1, c0, c1):
    return (pl.when((pl.col(c1) - pl.col(c0) == 1) & (pl.col(t1) - pl.col(t0) == 0)).then(pl.lit(CT))
            .when((pl.col(t1) - pl.col(t0) == 1) & (pl.col(c1) - pl.col(c0) == 0)).then(pl.lit(T))
            .otherwise(pl.lit(0)))


R = R.with_columns(
    side_expr("t_score", "t_next", "ct_score", "ct_next").alias("S"),
    side_expr("t_score_raw", "t_next_raw", "ct_score_raw", "ct_next_raw").alias("S_raw"),
)

re_all = read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count", "win_reason_code"])
re_counts = re_all.group_by(["match_key", "round"]).len().rename({"len": "n_round_end"})
re_ = re_all.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first").rename(
    {"tick": "end", "winner_team_code": "W"})
R = R.join(re_, on=["match_key", "round"], how="inner").join(re_counts, on=["match_key", "round"], how="left")

de = read("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([T, CT]))
dw = de.join(R.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner")
win = dw.filter((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
cnt = win.group_by(["match_key", "round"]).agg(
    pl.len().alias("n_deaths_win"),
    (pl.col("player_team_code") == T).sum().alias("n_T_deaths_win"),
    (pl.col("player_team_code") == CT).sum().alias("n_CT_deaths_win"),
    pl.col("tick").max().alias("last_tick"))
lastd = win.join(cnt.select(["match_key", "round", "last_tick"]), on=["match_key", "round"]).filter(
    pl.col("tick") == pl.col("last_tick")).group_by(["match_key", "round"]).agg(
    pl.col("player_team_code").n_unique().alias("n_victim_sides"),
    pl.col("player_team_code").min().alias("victim_side"))
R = R.join(cnt, on=["match_key", "round"], how="left").join(lastd, on=["match_key", "round"], how="left")
R = R.with_columns(
    (pl.col("end") - pl.col("last_tick")).alias("gap"),
    (pl.col("end") - pl.col("live")).alias("round_len"))
R = R.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("dd"))

bs = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
bo = bs.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(["match_key", "round"]).agg(
    pl.col("event_type").unique().alias("bomb_evs"), pl.col("tick").min().alias("bomb_out_tick"))
bp = bs.filter(pl.col("event_type") == "bomb_planted").group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("plant_tick"), pl.col("player_team_code").first().alias("planter_side"))
R = R.join(bo, on=["match_key", "round"], how="left").join(bp, on=["match_key", "round"], how="left")
R = R.with_columns(
    pl.when(pl.col("bomb_evs").list.contains("bomb_exploded")).then(pl.lit(T))
    .when(pl.col("bomb_evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(0)).alias("bomb_side"),
    pl.col("bomb_evs").list.len().fill_null(0).alias("n_bomb_out_types"))
R = R.with_columns(
    (pl.col("bomb_side") != 0).alias("has_bomb_out"),
    (pl.col("plant_tick").is_not_null()).alias("planted"))
out = SCRATCH / "verify_D4_6_rounds_2024.parquet"
R.write_parquet(out)
print("wrote", out, "rows", R.height)
print("matches", R["match_key"].n_unique(), "round_end dup rows (n_round_end>1):", R.filter(pl.col("n_round_end") > 1).height)
print("rows with S==0:", R.filter(pl.col("S") == 0).height, " rows with S_raw==0:", R.filter(pl.col("S_raw") == 0).height)
print("rows with no next freeze_end (S null):", R.filter(pl.col("t_next").is_null()).height)
print("player_count distribution (rows):")
print(R.group_by("player_count").len().sort("len", descending=True).head(6))
