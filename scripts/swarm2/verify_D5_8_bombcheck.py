r"""D5-8 verifier: for regime-A round-13 bomb rows, what do the engine reason, winner, bomb ticks and planter say?

Tests whether the bomb events for round 13 sit inside that round's tick window (end_tick of round 12 .. end_tick of 13),
and what the engine reason code is on the rows where bomb outcome disagrees with winner.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_8_bombcheck.py
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(50)

BASE = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\D5_8v")


def scan(ch):
    fd = os.path.join(BASE, f"basic_{ch}{SUF}")
    return pl.scan_parquet(sorted(glob.glob(os.path.join(fd, "dataframe_*"))), missing_columns="insert",
                           extra_columns="ignore")


R = pl.read_parquet(OUT / "rounds_v2.parquet")
bs = scan("bomb_state").select(["match_key", "round", "tick", "event_type", "player_team_code"]).collect()
rs_st = scan("round_state").select(["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["round_start", "round_freeze_end", "round_end"])).collect()

# round window ticks: start (round_start), freeze_end, end (round_end from round_state)
win = (rs_st.group_by(["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("t"))
       .pivot(on="event_type", index=["match_key", "round"], values="t"))
win = win.rename({c: f"w_{c}" for c in win.columns if c not in ("match_key", "round")})

ev = bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused", "bomb_planted"])).group_by(
    ["match_key", "round", "event_type"]).agg(pl.col("tick").min().alias("t")).pivot(
    on="event_type", index=["match_key", "round"], values="t")
ev = ev.rename({c: f"b_{c}" for c in ev.columns if c not in ("match_key", "round")})

J = (R.join(win, on=["match_key", "round"], how="left").join(ev, on=["match_key", "round"], how="left"))
print("pivot columns:", [c for c in J.columns if c.startswith("w_") or c.startswith("b_")])

# regime A late, round 13, bomb outcome present
A13 = J.filter((pl.col("grp") == "lateA") & (pl.col("round") == 13) & (pl.col("bside").is_not_null()))
print("\nlateA r13 bomb rows:", A13.height, " mismatch vs winner:", int((A13["winner"] != A13["bside"]).sum()))
print("reason codes on mismatch rows (winner != bomb side):")
mm = A13.filter(pl.col("winner") != pl.col("bside"))
print(mm.group_by(["reason", "winner", "bside"]).len().sort(["reason"]).rows())
print("reason codes on agreeing rows:")
print(A13.filter(pl.col("winner") == pl.col("bside")).group_by(["reason", "winner", "bside"]).len().rows())
print("player_count on mismatch rows:", mm.group_by("pc").len().sort("pc").rows())

# bomb tick position relative to the round window of the SAME round
def pos(df, col):
    return df.with_columns(
        inwin=((pl.col(col) >= pl.col("w_round_start")) & (pl.col(col) <= pl.col("w_round_end"))).fill_null(False))

for col in ["b_bomb_exploded", "b_bomb_defused", "b_bomb_planted"]:
    x = mm.join(win, on=["match_key", "round"], how="left")
    print(f"\n{col} present on mismatch rows: {int(mm[col].is_not_null().sum()) if col in mm.columns else 'n/a'}")

# compare the bomb tick with the round's own start/end window on all lateA r13 rows and on lateA r12/14 rows
def window_check(df, label):
    x = df.with_columns(
        bt=pl.coalesce(pl.col("b_bomb_exploded"), pl.col("b_bomb_defused")),
        inwin=(pl.coalesce(pl.col("b_bomb_exploded"), pl.col("b_bomb_defused")) >= pl.col("w_round_start"))
        & (pl.coalesce(pl.col("b_bomb_exploded"), pl.col("b_bomb_defused")) <= pl.col("w_round_end")))
    print(label, "rows:", x.height, " bomb tick inside round_start..round_end:", int(x["inwin"].sum()),
          " missing window:", int(x["w_round_end"].is_null().sum()))
    return x

x13 = window_check(J.filter((pl.col("grp") == "lateA") & (pl.col("round") == 13) & pl.col("bside").is_not_null()), "lateA r13")
x13m = x13.filter(pl.col("winner") != pl.col("bside"))
print("lateA r13 mismatch rows, bomb inside window:", int(x13m["inwin"].sum()), "of", x13m.height)
print("lateA r13 agree rows, bomb inside window:", int(x13.filter(pl.col("winner") == pl.col("bside"))["inwin"].sum()))
print(x13m.select(["match_key", "round", "reason", "winner", "bside", "w_round_start", "w_round_end", "w_freeze_end",
                   "b_bomb_planted", "b_bomb_exploded", "b_bomb_defused", "planter"]).head(12))
