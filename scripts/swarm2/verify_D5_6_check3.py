"""verify D5-6 check 3: a last-victim-free contradiction test per label and group.

For each round with no bomb plant, round_end winner_team_code (rw) is contradicted if every player of the
rw side died inside that round's freeze_end..round_end window (a side with all five players dead cannot be the
no-plant winner, ASSUMPTION: five players per side, distinct player_id). Also reports the share where the other
side is fully dead, and the chain agreement. Deaths are restricted to [fe, et] of their own label.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check3.py
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
T = T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & pl.col("grp").is_not_null())
NP = T.filter(pl.col("bs_planted") == 0).select(K + ["grp", "r1_end", "rw", "fe_tick", "re_tick", "chain"])

DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
D = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
per_side = (D.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n_dead"))
            .pivot(on="player_team_code", index=K, values="n_dead", aggregate_function="first")
            .rename({"2": "T_dead", "3": "CT_dead"}))
per_side = per_side.with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
X = NP.join(per_side, on=K, how="left").with_columns(
    T_dead=pl.col("T_dead").fill_null(0), CT_dead=pl.col("CT_dead").fill_null(0))
X = X.with_columns(
    rw_side_dead=pl.when(pl.col("rw") == 2).then(pl.col("T_dead") >= 5).otherwise(pl.col("CT_dead") >= 5),
    other_side_dead=pl.when(pl.col("rw") == 2).then(pl.col("CT_dead") >= 5).otherwise(pl.col("T_dead") >= 5),
    chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
    chain_dis=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
    .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
X = X.filter(pl.col("T_dead") + pl.col("CT_dead") > 0)  # rounds with at least one death in window

print("== C3-1: no-plant rounds with deaths: rw side fully dead (contradiction) vs other side fully dead ==")
t1 = X.group_by(["grp", "r1_end", "round"]).agg(
    pl.len().alias("n"),
    pl.col("rw_side_dead").mean().round(3).alias("rw_side_all5_dead"),
    pl.col("other_side_dead").mean().round(3).alias("other_side_all5_dead"),
    pl.col("chain_dis").mean().round(3).alias("chain_dis"),
    pl.col("chain_ok").sum().alias("n_chain_ok")).sort(["grp", "r1_end", "round"])
print(t1.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 28])))

print("\n== C3-2: label 13 regime A ctrl_late: contradiction by rw value and implied side ==")
c = X.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("round") == 13))
print(c.group_by("rw").agg(pl.len().alias("n"), pl.col("rw_side_dead").sum().alias("rw_side_all5_dead"),
                           pl.col("other_side_dead").sum().alias("other_side_all5_dead")).sort("rw"))

print("\n== C3-3: label 9 regime A ctrl_late: contradiction by rw value ==")
c9 = X.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("round") == 9))
print(c9.group_by("rw").agg(pl.len().alias("n"), pl.col("rw_side_dead").sum().alias("rw_side_all5_dead"),
                            pl.col("other_side_dead").sum().alias("other_side_all5_dead")).sort("rw"))
print("label 9 regime B / early contradiction:")
c9b = X.filter(pl.col("round") == 9).group_by(["grp", "r1_end"]).agg(
    pl.len().alias("n"), pl.col("rw_side_dead").mean().round(3).alias("rw_side_all5_dead"))
print(c9b.sort(["grp", "r1_end"]))

print("\n== C3-4: label-wise contradiction share across all labels, ctrl_late r1 true (for context) ==")
t4 = X.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end")).group_by("round").agg(
    pl.len().alias("n"), pl.col("rw_side_dead").mean().round(3).alias("rw_side_all5_dead")).sort("round")
print(t4.rows())
