"""verify_D7_8 part 3: triangulate player-based sides against bomb outcomes, planter sides and round 25 (stdout only)."""
import io
import contextlib
import glob
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_tri"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(260)
PI, H, XR, BO, RE, ST, BS, scan = (g[k] for k in ["PI", "H", "XR", "BO", "RE", "ST", "BS", "scan"])
RS = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"))

print("== T1. bomb outcome side vs PLAYER-BASED winner side (independent of winner_team_code)")
T = (XR.filter(pl.col("bomb_side").is_not_null() & pl.col("wside_p").is_not_null())
       .join(RS, on="match_key", how="left"))
T = T.with_columns((pl.col("bomb_side") == pl.col("wside_p")).alias("b_p_ok"),
                   (pl.col("bomb_side") == pl.col("winner_team_code")).alias("b_w_ok"))
print("rounds with bomb outcome and player-based winner side:", T.height)
print("by is_start x build (n, bomb==player-side count, bomb==winner_team_code count):")
print(T.group_by(["is_start", "build_num"]).agg(pl.len().alias("n"), pl.col("b_p_ok").sum().alias("bomb_eq_player"),
                                              pl.col("b_w_ok").sum().alias("bomb_eq_winner_code")).sort(["is_start", "build_num"]).rows())
print("player-side disagreements with bomb (rows):", T.filter(~pl.col("b_p_ok")).group_by(["is_start", "build_num", "round"]).len().sort("len", descending=True).head(10).rows())

print("\n== T2. planter cross-check: planter side from bomb_state vs player-based side on the planting round")
PLb = (scan("bomb_state").filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "player_id", "player_team_code"]).collect())
PLb = PLb.join(H.select(["match_key", "build_num"]), on="match_key", how="left").join(ST.with_columns(pl.lit(True).alias("is_start")), on=["match_key", "round"], how="left").with_columns(pl.col("is_start").fill_null(False))
PLb = PLb.join(PI.select(["match_key", "round", "player_id", "team_code"]), on=["match_key", "round", "player_id"], how="left")
print("planted rows by build x is_start x planter bomb_state team (count):")
print(PLb.group_by(["build_num", "is_start", "player_team_code"]).len().sort(["build_num", "is_start", "player_team_code"]).rows())
print("planter team per bomb_state != player_info team (rows, by build x is_start):")
print(PLb.filter(pl.col("team_code").is_not_null() & (pl.col("team_code") != pl.col("player_team_code"))).group_by(["build_num", "is_start"]).len().rows())

print("\n== T3. round 25 (first round of OT) and round 28 by build (winner-based, all counts)")
R25 = XR.filter(pl.col("round").is_in([24, 25, 26, 27, 28]) & pl.col("winner_team_code").is_not_null() & pl.col("wside_p").is_not_null())
print(R25.group_by(["round", "build_num"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm_winner"),
                                              pl.col("is_start").sum().alias("n_start")).sort(["round", "build_num"]).rows())
B25 = BO.join(RE.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="inner").join(H.select(["match_key", "build_num"]), on="match_key", how="left")
B25 = B25.filter(pl.col("round").is_in([24, 25, 26, 27, 28])).with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
print("bomb-based (round, build, n, mm):", B25.group_by(["round", "build_num"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["round", "build_num"]).rows())

print("\n== T4. header match_type and number_of_points by roster size")
HH = pl.concat([pl.read_parquet(f, columns=["key", "match_type", "number_of_points", "game", "tick_rate"]) for f in sorted(glob.glob(str(g["BASE"] / "header" / "dataframe_*")))]).rename({"key": "match_key"})
HH = HH.join(RS, on="match_key", how="left")
print("match_type x n_players:", HH.group_by(["match_type", "n_players"]).len().sort(["match_type", "n_players"]).rows())
print("game x tick_rate:", HH.group_by(["game", "tick_rate"]).len().rows())
