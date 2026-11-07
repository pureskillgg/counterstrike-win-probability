r"""Verifier for D5-5, stage 4.

  a. Per label x r1: among clean-step rounds without a bomb outcome and with an elimination-type
     round_end reason, which winner (step or round_end) is supported by the death channel?
     supported(T win) := all CT players spawned in the round are dead inside the window;
     supported(CT win) := all T players dead. (ASSUMPTION: elimination end implies roster wipe.)
  b. Label 9, r1-true: reasons and death support for the disagreeing rounds.
  c. Worked example 2024/02/12/m_161472e7ab8d0a97e6250fef: raw freeze_end / round_end / bomb rows.
  d. Label-13 bomb mismatches by folder day (csds/2024/02/DD) vs header match_date day.

Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_05_support.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
WIN = pl.read_parquet(SCR / "win_elim.parquet")
W = pl.read_parquet(SCR / "wide.parquet")
RS = pl.read_parquet(SCR / "round_state_sub.parquet")
BS = pl.read_parquet(SCR / "bomb_state_sub.parquet")
RE_raw = pl.read_parquet(SCR / "round_end.parquet")
H = pl.read_parquet(SCR / "header.parquet")
R1 = W.filter(pl.col("k") == 1).select(["match_key", pl.col("rw").is_not_null().alias("r1")])

# ---------- a. death support, per label x r1 ----------
A = WIN.filter((pl.col("step") != 0) & pl.col("rw").is_not_null() & pl.col("reason").is_in([8, 9])
               & pl.col("bomb_bs").is_null())
A = A.with_columns(
    step_sup=pl.when(pl.col("step") == 2).then(pl.col("allCT_dead")).otherwise(pl.col("allT_dead")),
    rw_sup=pl.when(pl.col("rw") == 2).then(pl.col("allCT_dead")).otherwise(pl.col("allT_dead")))
tab = (A.group_by(["k", "r1"]).agg(
    pl.len().alias("n"),
    (pl.col("step_sup") & ~pl.col("rw_sup")).sum().alias("step_only_supported"),
    (pl.col("rw_sup") & ~pl.col("step_sup")).sum().alias("rw_only_supported"),
    (pl.col("step_sup") & pl.col("rw_sup")).sum().alias("both"),
    (~pl.col("step_sup") & ~pl.col("rw_sup")).sum().alias("neither"))
    .sort(["k", "r1"]))
print("== a. no-bomb, elimination-coded rounds with clean step: death support by label x r1 ==")
print(tab.filter(pl.col("k") <= 30))

# ---------- b. label 9, r1-true disagreements ----------
B9 = W.filter((pl.col("k") == 9) & (pl.col("step") != 0) & pl.col("rw").is_not_null()).join(R1, on="match_key", how="left")
B9 = B9.with_columns(agree=pl.col("step") == pl.col("rw"))
print("\n== b. label 9 clean-step rows: agreement by r1 x reason x bomb ==")
print(B9.with_columns(bomb=pl.col("bomb_bs").fill_null(-9)).group_by(["r1", "agree", "reason", "bomb"]).agg(pl.len().alias("n"))
      .sort(["r1", "agree", "reason", "bomb"]))

# ---------- c. worked example ----------
MK = "csds/2024/02/12/m_161472e7ab8d0a97e6250fef/csds"
fe = (RS.filter((pl.col("match_key") == MK) & (pl.col("event_type") == "round_freeze_end"))
      .select(["round", "tick", "t_score", "ct_score"]).sort(["round", "tick"]))
print("\n== c. worked example: freeze_end rows (label, tick, t_score, ct_score) ==")
print(fe.filter(pl.col("round").is_between(11, 15)))
re = RE_raw.filter(pl.col("match_key") == MK).sort(["round", "tick"]).select(
    ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"])
print("round_end rows:")
print(re.filter(pl.col("round").is_between(11, 15)))
bo = BS.filter((pl.col("match_key") == MK) & pl.col("event_type").is_in(["bomb_planted", "bomb_exploded", "bomb_defused"])) \
    .select(["round", "tick", "event_type", "player_team_code"]).sort(["round", "tick"])
print("bomb_state rows:")
print(bo.filter(pl.col("round").is_between(11, 15)))
ex = WIN.filter((pl.col("match_key") == MK) & pl.col("k").is_between(12, 14)).select(
    ["k", "fe_tick", "re_tick", "rw", "reason", "step", "bomb_bs", "nT", "nCT", "dT", "dCT", "allT_dead", "allCT_dead"])
print("derived per-label window (fe_tick, re_tick, step, rw, reason, bomb, deaths by team, roster):")
print(ex.sort("k"))
exr = WIN.filter((pl.col("match_key") == MK)).select(["k", "fe_tick", "re_tick", "rw", "reason", "step", "bomb_bs"]).sort("k")
print(exr)
W_ex = W.filter((pl.col("match_key") == MK) & pl.col("k").is_between(12, 14)).select(
    ["k", "t", "ct", "fe_tick", "re_tick", "plant_tick", "b_tick", "rw", "reason", "n_plant", "nT", "nCT"])
print(W_ex.sort("k"))

# ---------- d. label-13 mismatch by folder day vs match_date day ----------
L13 = W.filter((pl.col("k") == 13) & pl.col("rw").is_not_null() & pl.col("bomb_bs").is_not_null() & (pl.col("bomb_bs") >= 0))
L13 = L13.with_columns(mm=pl.col("bomb_bs") != pl.col("rw"))
L13 = L13.with_columns(mdate_day=pl.col("match_date").str.slice(8, 2))
print("\n== d. label 13 bomb-outcome rows, mismatch rate by folder day (csds/2024/02/DD) ==")
print(L13.group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("day"))
print("\n== d2. same by header match_date day ==")
print(L13.group_by("mdate_day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("mdate_day"))
print("\n== d3. folder day vs match_date day agreement (all matches) ==")
print(H.with_columns(md=pl.col("match_date").str.slice(8, 2)).group_by(
    (pl.col("day") == pl.col("md"))).len())
print("\nlabel-13 bomb rows total:", L13.height, " mismatches:", int(L13["mm"].sum()),
      " in folder days 12..25:", int(L13.filter(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)]))["mm"].sum()),
      " rows in days 12..25:", L13.filter(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).height)
