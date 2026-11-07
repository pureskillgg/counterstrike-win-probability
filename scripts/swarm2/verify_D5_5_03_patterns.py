r"""Verifier for D5-5, stage 2: where do round_end winners disagree with the freeze-end step?

Uses scratch extracts from verify_D5_5_01_load.py and the wide table from verify_D5_5_02_checks.py.
Adds the player_death channel (tick-window elimination test, independent of round_end and of the
score step) and a swap-aware step for label 12 (score-only, no round_end input).
ASSUMPTIONS: window = [freeze_end tick, round_end tick]; roster = distinct players spawned for the team
in that round; 'all dead' = distinct victims >= roster size.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_03_patterns.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")


def scan(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f) for f in fs], how="vertical_relaxed")


W = pl.read_parquet(SCR / "wide.parquet")
RS = pl.read_parquet(SCR / "round_state_sub.parquet")
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("tick").first().alias("fe_tick")))
W = W.with_columns(pl.col("k").alias("k_"))
R1 = W.filter(pl.col("k") == 1).select(["match_key", pl.col("rw").is_not_null().alias("r1")])
W = W.join(R1, on="match_key", how="left").with_columns(pl.col("r1").fill_null(False))

# ---------- 1. naive-step disagreement by label x r1 ----------
print("== 1. clean naive step vs round_end winner, by label and r1 (round_end label 1 present) ==")
C = W.filter((pl.col("step") != 0) & pl.col("rw").is_not_null())
C = C.with_columns(agree=pl.col("step") == pl.col("rw"))
tab = (C.group_by(["k", "r1"]).agg(pl.len().alias("clean"), pl.col("agree").sum().alias("agree"))
       .with_columns((pl.col("clean") - pl.col("agree")).alias("disagree"))
       .sort(["k", "r1"]))
print(tab.filter(pl.col("disagree") > 0))

# ---------- 2. label-13 reason cross-tab ----------
L13 = W.filter((pl.col("k") == 13) & pl.col("rw").is_not_null())
print("\n== 2. label 13: round_end (winner, reason) by r1 ==")
print(L13.group_by(["r1", "rw", "reason"]).agg(pl.len().alias("n")).sort(["r1", "rw", "reason"]))
print("\n== 2b. label 13 r1-true clean-step rows: (step, rw, reason, bomb) ==")
Lc = L13.filter(pl.col("r1") & (pl.col("step") != 0))
print(Lc.group_by(["step", "rw", "reason", pl.col("bomb_bs").fill_null(-9).alias("bomb_bs")]).agg(pl.len().alias("n"))
      .sort(["step", "rw", "reason", "bomb_bs"]))

# ---------- 3. death-based elimination test ----------
DE = (scan("player_death").select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
      .filter(pl.col("player_team_code").is_in([2, 3])))
WIN = (W.select(["match_key", "k", "fe_tick", "re_tick", "nT", "nCT", "rw", "reason", "step", "bomb_bs", "r1"])
       .filter(pl.col("fe_tick").is_not_null() & pl.col("re_tick").is_not_null()))
J = WIN.join(DE.rename({"player_team_code": "vteam"}), on="match_key", how="inner")
J = J.filter((pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
D = (J.group_by(["match_key", "k", "vteam"]).agg(pl.col("player_id").n_unique().alias("ndead"))
     .pivot(on="vteam", index=["match_key", "k"], values="ndead").rename({"2": "dT", "3": "dCT"}))
WIN = WIN.join(D, on=["match_key", "k"], how="left").with_columns(
    dT=pl.col("dT").fill_null(0), dCT=pl.col("dCT").fill_null(0))
WIN = WIN.with_columns(
    allT_dead=(pl.col("dT") >= pl.col("nT")) & pl.col("nT").is_not_null(),
    allCT_dead=(pl.col("dCT") >= pl.col("nCT")) & pl.col("nCT").is_not_null())
# expectation: reason 8 (CTs win) -> all T dead; reason 9 (Ts win) -> all CT dead
E = WIN.filter(pl.col("reason").is_in([8, 9]) & pl.col("rw").is_not_null())
E = E.with_columns(
    elim_ok=pl.when(pl.col("reason") == 8).then(pl.col("allT_dead")).otherwise(pl.col("allCT_dead")))
print("\n== 3. elimination consistency for round_end reasons 8/9 (all labels), by label group ==")
print(E.with_columns(grp=pl.when(pl.col("k") == 13).then(pl.lit("k13")).otherwise(pl.lit("other")))
      .group_by(["grp", "r1", "reason", "elim_ok"]).agg(pl.len().alias("n")).sort(["grp", "r1", "reason", "elim_ok"]))

# step-based elimination test at label 13: step says team X won; if elimination, the other side is all dead
S13 = WIN.filter((pl.col("k") == 13) & (pl.col("step") != 0) & pl.col("rw").is_not_null())
S13 = S13.with_columns(
    step_side_dead=pl.when(pl.col("step") == 2).then(pl.col("allCT_dead")).otherwise(pl.col("allT_dead")),
    rw_side_dead=pl.when(pl.col("rw") == 2).then(pl.col("allCT_dead")).otherwise(pl.col("allT_dead")),
    rw_is_elim=pl.col("reason").is_in([8, 9]),
    bomb_present=pl.col("bomb_bs").is_not_null())
print("\n== 3b. label 13 clean-step rows: elimination evidence vs winners ==")
print(S13.group_by(["r1", "bomb_present", "rw_is_elim", "step", "rw", "allT_dead", "allCT_dead"]).agg(pl.len().alias("n"))
      .sort(["r1", "bomb_present", "rw_is_elim", "step", "rw", "allT_dead", "allCT_dead"]))

# planted-bomb & elimination reason (rules: elimination end with planted bomb is not expected)
print("\n== 3c. rounds with a planted bomb: round_end reason by label group ==")
P = W.filter(pl.col("n_plant").is_not_null() & pl.col("rw").is_not_null())
print(P.with_columns(grp=pl.when(pl.col("k") == 13).then(pl.lit("k13")).otherwise(pl.lit("other")))
      .group_by(["grp", "r1", "reason"]).agg(pl.len().alias("n")).sort(["grp", "r1", "reason"]))

WIN.write_parquet(SCR / "win_elim.parquet")
