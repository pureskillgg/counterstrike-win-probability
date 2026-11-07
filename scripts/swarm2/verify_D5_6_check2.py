"""verify D5-6 check 2: split label-13 disagreement by plant / outcome status, test label 9, test the override.

Uses scratchpad parquet from verify_D5_6_build.py and victim-death rows re-read from the 2024 tome on G:.
Truths: chain (freeze-end score step, valid when no side swap between fe_r and fe_{r+1}), implied (last victim),
bomb outcome (bomb_state defused/exploded). rw = round_end winner_team_code.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check2.py
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
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
M = M.with_columns(grp=pl.when(pl.col("aff_pc20")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late"))
                   .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
T = T.join(M.select(["match_key", "grp", "r1_end"]), on="match_key", how="left")
base = T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & pl.col("grp").is_not_null())
base = base.with_columns(
    status=pl.when(pl.col("bs_planted") == 0).then(pl.lit("no_plant"))
    .when(pl.col("bomb_side").is_not_null()).then(pl.lit("plant_outcome"))
    .otherwise(pl.lit("plant_no_outcome")),
    chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0))
base = base.with_columns(
    chain_dis=pl.when(pl.col("chain_ok")).then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None),
    imp_dis=pl.when(pl.col("implied").is_not_null()).then((pl.col("implied") != pl.col("rw")).cast(pl.Int8)).otherwise(None))

print("== C1: label 12, 13, 14 by group, r1_end and plant status: rw disagreement with chain and with implied ==")
c1 = base.filter(pl.col("round").is_in([12, 13, 14])).group_by(["grp", "r1_end", "round", "status"]).agg(
    pl.len().alias("n"),
    pl.col("chain_ok").sum().alias("n_chain_ok"),
    pl.col("chain_dis").mean().round(3).alias("chain_dis_rate"),
    pl.col("implied").is_not_null().sum().alias("n_imp"),
    pl.col("imp_dis").mean().round(3).alias("imp_dis_rate"),
).sort(["grp", "r1_end", "round", "status"])
print(c1)

print("\n== C2: label 13, ctrl_late r1=true: outcome presence vs rw (bomb side) for plant rows ==")
c2 = base.filter((pl.col("round") == 13) & (pl.col("grp") == "ctrl_late") & pl.col("r1_end"))
print(c2.group_by(["status", "rw"]).agg(pl.len().alias("n"), pl.col("chain_ok").sum().alias("chain_ok"),
                                         pl.col("chain_dis").sum().alias("chain_dis")).sort(["status", "rw"]))

print("\n== C3: label 13 ctrl_late r1=true no-plant rounds: rw distribution vs implied and chain ==")
c3 = c2.filter(pl.col("status") == "no_plant").filter(pl.col("implied").is_not_null())
print("rows:", c3.height, " rw==2 (T win):", int((c3["rw"] == 2).sum()), " implied==2:", int((c3["implied"] == 2).sum()),
      " chain==2:", int((c3["chain"] == 2).sum()), " chain valid:", int(c3["chain_ok"].sum()))
print(c3.group_by(["rw", "implied"]).agg(pl.len().alias("n")).sort(["rw", "implied"]))

print("\n== C4: label 9 ctrl_late r1=true: crosstab rw x implied and neighbour test ==")
nbA = base.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end")).select(K + ["rw", "implied", "chain", "bs_planted"])
l9 = nbA.filter((pl.col("round") == 9) & (pl.col("bs_planted") == 0) & pl.col("implied").is_not_null())
print("label 9 no-plant rows:", l9.height)
print(l9.group_by(["implied", "rw"]).agg(pl.len().alias("n")).sort(["implied", "rw"]))
for k in [-1, 1]:
    nb = nbA.select(["match_key", "round", "rw"]).rename({"rw": "rw_n"}).with_columns(round=pl.col("round") - k)
    j = l9.join(nb, on=["match_key", "round"], how="left")
    print(f"implied9 == rw(9{k:+d}):", int((j["implied"] == j["rw_n"]).sum()), "of", int(j["rw_n"].is_not_null().sum()))

print("\n== C5: label 9 ctrl_late r1=true, outcome-based rows (bomb) for label 9 ==")
print(base.filter((pl.col("round") == 9) & (pl.col("grp") == "ctrl_late") & pl.col("r1_end")).group_by(
    ["status"]).agg(pl.len().alias("n"), pl.col("chain_dis").mean().round(3).alias("chain_dis"),
                    pl.col("imp_dis").mean().round(3).alias("imp_dis")).sort("status"))

# ---- C6: distinct victims per side and timing of the last death vs round_end, label 13 ctrl A no-plant ----
DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3])).collect()
ids13 = c3.select(K + ["rw", "implied", "re_tick", "fe_tick"])
d13 = DE.join(ids13, on=K, how="inner")
dist = d13.group_by(K + ["rw", "implied"]).agg(
    pl.col("player_id").filter(pl.col("player_team_code") == 2).n_unique().alias("n_T_victims"),
    pl.col("player_id").filter(pl.col("player_team_code") == 3).n_unique().alias("n_CT_victims"),
    pl.col("tick").max().alias("last_tick"),
    pl.col("re_tick").first().alias("re_tick"),
    pl.col("fe_tick").first().alias("fe_tick"),
    (pl.col("tick") > pl.col("re_tick")).sum().alias("n_after_end"),
    (pl.col("tick") < pl.col("fe_tick")).sum().alias("n_before_fe"))
dist = dist.with_columns(gap_last_to_end=pl.col("re_tick") - pl.col("last_tick"))
print("\n== C6: label 13 ctrl A no-plant: victims per side, deaths after round_end, timing (sample per rw/implied) ==")
print(dist.group_by(["rw", "implied"]).agg(
    pl.len().alias("n"), pl.col("n_CT_victims").mean().round(2).alias("mean_CT_victims"),
    pl.col("n_T_victims").mean().round(2).alias("mean_T_victims"),
    (pl.col("n_CT_victims") == 5).mean().round(3).alias("share_all5_CT_dead"),
    (pl.col("n_T_victims") == 5).mean().round(3).alias("share_all5_T_dead"),
    (pl.col("n_after_end") > 0).mean().round(3).alias("share_with_death_after_end"),
    pl.col("gap_last_to_end").median().alias("median_gap_last_death_to_end_ticks")).sort(["rw", "implied"]))

# ---- C7: third-source override: how many rw would change under a 3-source majority (rw, chain, implied) ----
print("\n== C7: 3-source majority override (rw, chain, implied) on no-plant rounds with deaths, all labels, ctrl A ==")
ov = base.filter((pl.col("status") == "no_plant") & pl.col("implied").is_not_null() & pl.col("chain_ok"))
ov = ov.with_columns(maj_override=(pl.col("chain") == pl.col("implied")) & (pl.col("implied") != pl.col("rw")))
print(ov.group_by(["grp", "r1_end"]).agg(
    pl.len().alias("n_three_sources"), pl.col("maj_override").sum().alias("rw_overridden")).sort(["grp", "r1_end"]))
print("overridden rows by label (ctrl_late r1 true):")
print(ov.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & pl.col("maj_override")).group_by("round").agg(
    pl.len().alias("n")).sort("round").rows())
