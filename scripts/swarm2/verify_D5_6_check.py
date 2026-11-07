"""verify D5-6 check: recompute the claimed numbers and test the strongest alternatives.

Inputs: scratchpad parquet written by verify_D5_6_build.py (per match-round and per match), plus
player_death and bomb_state rows re-read from the 2024 tome on G: with projection.
Truths compared with round_end winner_team_code (rw):
  implied   last victim side opposite (player_death, max tick; ties excluded)   -- the claim's truth
  chain     freeze_end score step fe_r -> fe_{r+1} (valid only if no side swap between them)
  bomb      bomb outcome side from bomb_state (defused -> 3, exploded -> 2)   -- no side codes used
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(50)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_6")
K = ["match_key", "round"]


def scan(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.scan_parquet(fs).select(cols)


T = pl.read_parquet(SCR / "v_rounds.parquet")
M = pl.read_parquet(SCR / "v_matches.parquet")
hdr = (pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
       .rename({"key": "match_key"}).with_columns(
           final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")))
M = M.join(hdr.select(["match_key", "final_sum"]), on="match_key", how="left")

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_pc20")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
T = T.join(M.select(["match_key", "grp", "r1_end", "final_sum", "day"]), on="match_key", how="left")

# death-level victim sides (for the window check and planter check)
DE = scan("player_death", ["match_key", "round", "tick", "player_team_code"]).filter(
    pl.col("player_team_code").is_in([2, 3])).collect()
BSP = (scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"]).filter(
    pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect())

# ---- Section 0: structure of round labels per match (fe/et labels vs header final score) ----
print("== S0: label range of round_end (et) per match, by regime, vs total rounds from header final score ==")
rng = (T.filter(pl.col("re_present")).group_by("match_key").agg(
    pl.col("round").min().alias("min_lab"), pl.col("round").max().alias("max_lab"), pl.len().alias("n_lab")))
rng = rng.join(M.select(["match_key", "grp", "r1_end", "final_sum"]), on="match_key", how="left")
rng = rng.with_columns(
    minlab_is_1=pl.col("min_lab") == 1,
    n_eq_final=(pl.col("n_lab") == pl.col("final_sum")),
    maxlab_eq_final=(pl.col("max_lab") == pl.col("final_sum")),
    n_plus1_final=(pl.col("max_lab") == pl.col("final_sum") + 1))
print(rng.group_by(["grp", "r1_end", "minlab_is_1"]).agg(
    pl.len().alias("matches"), pl.col("n_eq_final").sum().alias("n_labels_eq_final"),
    pl.col("maxlab_eq_final").sum().alias("maxlab_eq_final"), pl.col("n_plus1_final").sum().alias("maxlab_eq_final_plus1"))
      .sort(["grp", "r1_end", "minlab_is_1"]))

# ---- Section 1: reproduce the claimed numbers (D5_06 / D5_08 definitions) ----
base = T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & pl.col("grp").is_not_null())
nb = base.filter((pl.col("bs_planted") == 0) & pl.col("implied").is_not_null())
nb = nb.with_columns(
    dis_rw=(pl.col("implied") != pl.col("rw")).cast(pl.Int8),
    chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
    ch_eq_imp=(pl.col("chain") == pl.col("implied")),
    ch_eq_rw=(pl.col("chain") == pl.col("rw")))
print("\n== S1: no-bomb rounds with deaths, implied vs rw (dis = disagreement) by group, r1_end, label ==")
rep = nb.filter(pl.col("round").is_in([9, 12, 13, 14, 25, 28])).group_by(["grp", "r1_end", "round"]).agg(
    pl.len().alias("n"), pl.col("dis_rw").sum().alias("dis_rw"),
    (pl.col("dis_rw").sum() / pl.len()).round(3).alias("rate_dis"),
    pl.col("chain_ok").sum().alias("chain_ok_n"),
    pl.col("ch_eq_imp").sum().alias("chain_eq_implied")).sort(["grp", "r1_end", "round"])
print(rep)
print("\nCLAIM check: ctrl_late r1_end=true label 13 -> claim n=945, dis=769 (81%), chain agrees w/ implied 870")
c13 = nb.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("round") == 13))
print("recomputed: n =", c13.height, " dis_rw =", int(c13["dis_rw"].sum()),
      " rate =", round(c13["dis_rw"].mean(), 4),
      " chain_eq_implied(all rows incl. chain null as False) =", int(c13["ch_eq_imp"].sum()),
      " chain_ok rows =", int(c13["chain_ok"].sum()))

# ---- Section 2: confusion of implied vs rw vs chain at label 13, ctrl_late r1 true ----
print("\n== S2: label 13, ctrl_late, r1_end=true, no-bomb with deaths: crosstabs ==")
print("implied x rw:")
print(c13.group_by(["implied", "rw"]).agg(pl.len().alias("n")).sort(["implied", "rw"]))
print("chain x rw (chain null excluded):")
print(c13.filter(pl.col("chain").is_not_null()).group_by(["chain", "rw"]).agg(pl.len().alias("n")).sort(["chain", "rw"]))
print("chain x implied (chain null excluded):")
print(c13.filter(pl.col("chain").is_not_null()).group_by(["chain", "implied"]).agg(pl.len().alias("n")).sort(["chain", "implied"]))
print("reason code of disagreeing rows (rw_reason):")
print(c13.filter(pl.col("dis_rw") == 1).group_by(["rw_reason", "rw"]).agg(pl.len().alias("n")).sort(["rw_reason", "rw"]))

# ---- Section 3: bomb truth (independent of deaths, chain and side codes) at label 13 ----
bo = base.filter(pl.col("bomb_side").is_not_null())
bo = bo.with_columns(bomb_dis=(pl.col("bomb_side") != pl.col("rw")).cast(pl.Int8),
                     chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     ch_eq_bomb=(pl.col("chain") == pl.col("bomb_side")),
                     ch_eq_rw=(pl.col("chain") == pl.col("rw")))
print("\n== S3: rounds with a bomb outcome, label 13: bomb vs rw and chain, by group ==")
print(bo.filter(pl.col("round") == 13).group_by(["grp", "r1_end"]).agg(
    pl.len().alias("n"), pl.col("bomb_dis").sum().alias("bomb_dis"),
    pl.col("chain_ok").sum().alias("chain_ok"), pl.col("ch_eq_bomb").sum().alias("chain_eq_bomb"),
    pl.col("ch_eq_rw").sum().alias("chain_eq_rw")).sort(["grp", "r1_end"]))
print("bomb outcome x rw, label 13, aff group:")
print(bo.filter((pl.col("round") == 13) & (pl.col("grp") == "aff_r13")).group_by(["bomb_side", "rw", "rw_reason"]).agg(
    pl.len().alias("n")).sort(["bomb_side", "rw", "rw_reason"]))
print("bomb outcome x rw, all other labels (regime A and B, all groups) for contrast:")
print(bo.filter(pl.col("round") != 13).group_by(["bomb_dis"]).agg(pl.len().alias("n")))
print("bomb outcome x rw, labels 12 and 14, regime A ctrl_late:")
print(bo.filter(pl.col("round").is_in([12, 14]) & (pl.col("grp") == "ctrl_late") & pl.col("r1_end")).group_by(
    ["round", "bomb_dis"]).agg(pl.len().alias("n")).sort(["round", "bomb_dis"]))

# ---- Section 4: planter side at label 13 (bomb_planted player_team_code should be 2 = T) ----
PL = BSP.filter(pl.col("event_type") == "bomb_planted").select(K + ["player_team_code"]).unique(subset=K, keep="first")
PL = PL.join(T.select(K + ["grp", "r1_end"]), on=K, how="left")
print("\n== S4: planter side (bomb_planted player_team_code) by label and group; expected 2 (T) ==")
print(PL.filter(pl.col("round").is_in([12, 13, 14]) & pl.col("grp").is_not_null()).group_by(
    ["grp", "r1_end", "round", "player_team_code"]).agg(pl.len().alias("n")).sort(["grp", "r1_end", "round", "player_team_code"]))

# ---- Section 5: window check, deaths inside [fe_r, et_r] of their own label ----
Dw = DE.join(T.select(K + ["fe", "et"]) if "fe" in T.columns else T.select(K + ["fe_tick", "re_tick"]).rename(
    {"fe_tick": "fe", "re_tick": "et"}), on=K, how="left")
Dw = Dw.with_columns(inwin=((pl.col("tick") >= pl.col("fe")) & (pl.col("tick") <= pl.col("et"))))
Dw = Dw.join(T.select(K + ["grp", "r1_end"]), on=K, how="left")
print("\n== S5: victim deaths inside [freeze_end_r, round_end_r] of their own label (share) ==")
print(Dw.filter(pl.col("round").is_in([12, 13, 14]) & pl.col("grp").is_not_null()).group_by(
    ["grp", "r1_end", "round"]).agg(pl.len().alias("n_deaths"), pl.col("inwin").mean().round(4).alias("share_inwin"))
      .sort(["grp", "r1_end", "round"]))

# ---- Section 6: neighbour-label test, regime A ctrl no-bomb label 13 ----
nbA = T.filter(pl.col("grp").is_in(["ctrl_late"]) & pl.col("r1_end") & pl.col("re_present"))
nbA = nbA.select(K + ["rw", "implied", "chain", "bs_planted"])
nbA13 = nbA.filter((pl.col("round") == 13) & (pl.col("bs_planted") == 0) & pl.col("implied").is_not_null())
nbA13 = nbA13.join(nbA.select(["match_key", "round", "rw"]).rename({"rw": "rw_m1"}).with_columns(round=pl.col("round") + 1)
                   , on=["match_key", "round"], how="left")
nbA13 = nbA13.join(nbA.select(["match_key", "round", "rw"]).rename({"rw": "rw_p1"}).with_columns(round=pl.col("round") - 1)
                   , on=["match_key", "round"], how="left")
print("\n== S6: neighbour labels, regime A ctrl no-bomb label 13 rows (n with deaths) ==")
print("n =", nbA13.height)
print("implied13 == rw12:", int((nbA13["implied"] == nbA13["rw_m1"]).sum()), " of", int(nbA13["rw_m1"].is_not_null().sum()))
print("implied13 == rw14:", int((nbA13["implied"] == nbA13["rw_p1"]).sum()), " of", int(nbA13["rw_p1"].is_not_null().sum()))
print("rw13 == rw12:", int((nbA13["rw"] == nbA13["rw_m1"]).sum()), " rw13 == rw14:", int((nbA13["rw"] == nbA13["rw_p1"]).sum()))

# ---- Section 7: timeouts and ties among rows used as truth ----
print("\n== S7: tie rows (ambiguous last victim) and reason of no-bomb rounds, label 13 ctrl A vs all ==")
tie = T.filter(pl.col("n_side_at_max") > 1)
print("tie rows (both victim sides at max tick) by label:", tie.group_by("round").len().sort("round").head(30).rows())
print("reason codes, no-bomb label 13 ctrl A:")
print(c13.group_by("rw_reason").agg(pl.len().alias("n"), pl.col("dis_rw").sum().alias("dis")).sort("rw_reason"))
print("reason codes, no-bomb rounds all other labels (ctrl_late, r1 true):")
oth = nb.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("round") != 13))
print(oth.group_by("rw_reason").agg(pl.len().alias("n"), pl.col("dis_rw").sum().alias("dis")).sort("rw_reason"))

# ---- Section 8: day split of label 13 disagreement, ctrl A ----
print("\n== S8: label 13 ctrl A no-bomb disagreement by day (n, dis, rate) ==")
print(c13.group_by("day").agg(pl.len().alias("n"), pl.col("dis_rw").sum().alias("dis"),
                             pl.col("dis_rw").mean().round(3).alias("rate")).sort("day"))
