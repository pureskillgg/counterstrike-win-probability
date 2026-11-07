"""verify_D7_8 part 2: roster size, all-player-count mismatch tables, day-vs-build share, planter cross-check (stdout only)."""
import io
import contextlib
import sys
from pathlib import Path

import polars as pl

src = Path(r"G:\pureskill\csds-cs2\scripts\swarm2\verify_D7_8_main.py").read_text(encoding="utf-8")
src = src.replace('sys.stdout.reconfigure(encoding="utf-8")', "pass")
g = {"__name__": "verify_main_roster"}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(src, "verify_D7_8_main.py", "exec"), g)
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(260)
PI, H, PM, XR, ST, BR, BS, BO, RE, starts = (g[k] for k in ["PI", "H", "PM", "XR", "ST", "BR", "BS", "BO", "RE", "starts"])

print("== 1. roster size (distinct player_id with team 2/3 in player_info) per match")
RS = PI.group_by("match_key").agg(pl.col("player_id").n_unique().alias("n_players"), pl.col("round").n_unique().alias("n_rounds_pi"))
print("n_players distribution:", RS.group_by("n_players").len().sort("len", descending=True).head(10).rows())
M = PM.join(RS, on="match_key", how="left").join(H.select(["match_key", "build_num"]), on="match_key", how="left", suffix="_h")
print("fmt x n_players (matches):", M.group_by(["fmt", "n_players"]).len().sort(["fmt", "n_players"]).rows())
print("fmt x build x n_players (matches):", M.group_by(["fmt", "build_num", "n_players"]).len().sort(["fmt", "build_num", "n_players"]).rows())
rc = (RE.join(H.select(["match_key"]), on="match_key").group_by("match_key")
        .agg(pl.col("player_count").mode().first().alias("pc_mode")))
print("round_end player_count modal per match x n_players (top):", rc.join(RS, on="match_key").group_by(["pc_mode", "n_players"]).len().sort("len", descending=True).head(10).rows())

print("\n== 2. winner-based mismatch, ALL player_count values (not only 20), by roster and format")
Q_all = XR.filter(pl.col("winner_team_code").is_not_null() & pl.col("wside_p").is_not_null())
Q_all = Q_all.join(RS, on="match_key", how="left")
print("rows:", Q_all.height, "| mismatches:", Q_all.filter(pl.col("w_p_ok") == False).height)
print("by n_players x is_start (n, mm):")
print(Q_all.group_by(["n_players", "is_start"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm")).sort(["n_players", "is_start"]).rows())
print("is_start rows by fmt x build x n_players (n, mm):")
print(Q_all.filter(pl.col("is_start")).group_by(["fmt", "build_num", "n_players"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm")).sort(["fmt", "build_num", "n_players"]).rows())
print("non-start rows: mismatches by build x n_players:")
print(Q_all.filter(~pl.col("is_start")).group_by(["build_num", "n_players"]).agg(pl.len().alias("n"), (pl.col("w_p_ok") == False).sum().alias("mm")).sort(["build_num", "n_players"]).rows())

print("\n== 3. bomb-based defect check, all player_count values, by build x is_start (n, mm)")
BA = (RE.join(BO.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="inner")
        .join(H.select(["match_key", "day", "build_num"]), on="match_key", how="left")
        .join(ST.with_columns(pl.lit(True).alias("is_start_b")), on=["match_key", "round"], how="left")
        .with_columns(pl.col("is_start_b").fill_null(False), (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
        .join(RS, on="match_key", how="left"))
print("bomb rounds all player_count:", BA.height, "| mismatches:", BA.filter(pl.col("mm")).height)
print(BA.group_by(["build_num", "is_start_b"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "is_start_b"]).rows())
print("bomb mismatches by n_players x build x is_start:", BA.filter(pl.col("mm")).group_by(["n_players", "build_num", "is_start_b"]).len().sort(["n_players", "build_num", "is_start_b"]).rows())
print("bomb rounds by n_players x build x is_start (n):", BA.group_by(["n_players", "build_num", "is_start_b"]).agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["n_players", "build_num", "is_start_b"]).rows())

print("\n== 4. day gradient vs build share (matches) and is_start mismatch rate (clean, player_count 20, f12 starts)")
share = (H.group_by("day").agg((pl.col("build_num") == 9961).mean().alias("share_9961"), pl.len().alias("n_matches")).sort("day"))
Qc = XR.filter((pl.col("player_count") == 20) & pl.col("winner_team_code").is_not_null() & pl.col("wside_p").is_not_null() & pl.col("is_start") & (pl.col("fmt") == "f12"))
rate = Qc.group_by("day").agg(pl.len().alias("n_start"), (pl.col("w_p_ok") == False).mean().alias("mm_rate")).sort("day")
print(share.join(rate, on="day", how="left").with_columns(pl.col("share_9961").round(3), pl.col("mm_rate").round(3)).rows())

print("\n== 5. planter cross-check: bomb_state player_team_code vs player_info team_code for same player and round")
PLR = (BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "player_team_code"]))
PLR_pid = BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "player_id", "player_team_code"]) if "player_id" in BS.columns else None
import polars as _p
BSraw = (g["scan"]("bomb_state").filter(pl.col("event_type") == "bomb_planted")
         .select(["match_key", "round", "player_id", "player_team_code"]).collect())
pc = BSraw.join(PI.select(["match_key", "round", "player_id", "team_code"]), on=["match_key", "round", "player_id"], how="left")
print("planted rows:", BSraw.height, "| with player_info match:", pc.filter(pl.col("team_code").is_not_null()).height)
print("agree (bomb player_team_code == player_info team_code):", pc.filter(pl.col("team_code").is_not_null() & (pl.col("team_code") == pl.col("player_team_code"))).height,
      "| disagree:", pc.filter(pl.col("team_code").is_not_null() & (pl.col("team_code") != pl.col("player_team_code"))).height)
print("planter player_info team_code distribution:", pc.group_by("team_code").len().sort("team_code").rows())
