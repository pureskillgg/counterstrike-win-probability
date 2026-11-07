"""verify D5-6 check 4: completeness baseline for the all-five-dead test, and label 9 vs label 13 co-occurrence.

Rule used (ASSUMPTION: five players per side, no-bomb round): a round_end T win with no bomb plant requires all five
CT players to be dead in the round window; a CT win with no plant needs no T death pattern (timeouts are legal).
Baseline: share of no-plant rounds where CT is fully dead given rw=T, per label and regime. Clean labels should be
near the completeness ceiling; label 13 regime A should be near zero if the round_end winner is inverted.
Co-occurrence: per match, chain-based inversion flags at labels 9 and 13 (chain valid, no plant), regime A ctrl.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check4.py
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
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
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

# completeness of death rows: null side / null id, per channel
DALL = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).collect()
print("player_death rows:", DALL.height,
      " null side:", DALL.filter(pl.col("player_team_code").is_null()).height,
      " null player_id:", DALL.filter(pl.col("player_id").is_null()).height,
      " null side or id:", DALL.filter(pl.col("player_team_code").is_null() | pl.col("player_id").is_null()).height)

DE = DALL.filter(pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null())
NP = T.filter(pl.col("bs_planted") == 0).select(K + ["grp", "r1_end", "rw", "fe_tick", "re_tick", "chain", "implied"])
D = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (D.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
X = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
X = X.with_columns(
    chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
    chain_dis=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
    .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
X = X.filter((pl.col("T_dead") + pl.col("CT_dead")) > 0)

print("\n== C4-1: no-plant rounds with deaths. Given rw, share with the side that must be dead fully dead ==")
print("rw=T needs CT all dead (elimination). rw=CT: CT all dead is a contradiction; T all dead is consistent.")
b = X.group_by(["grp", "r1_end", "round"]).agg(
    pl.len().alias("n"),
    pl.col("rw").eq(2).sum().alias("n_rwT"),
    (pl.col("rw").eq(2) & (pl.col("CT_dead") >= 5)).sum().alias("rwT_CT_all5"),
    pl.col("rw").eq(3).sum().alias("n_rwCT"),
    (pl.col("rw").eq(3) & (pl.col("CT_dead") >= 5)).sum().alias("rwCT_CT_all5"),
    (pl.col("rw").eq(3) & (pl.col("T_dead") >= 5)).sum().alias("rwCT_T_all5"),
    pl.col("chain_dis").mean().round(3).alias("chain_dis"))
b = b.with_columns(
    rwT_CT_all5_share=(pl.col("rwT_CT_all5") / pl.col("n_rwT")).round(3),
    rwCT_CT_all5_share=(pl.col("rwCT_CT_all5") / pl.col("n_rwCT")).round(3)).sort(["grp", "r1_end", "round"])
print(b.filter(pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 25, 28])))

print("\n== C4-2: label 9 ctrl_late r1=true, rw=T rows: CT all dead vs CT alive, and implied/chain ==")
c9 = X.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("round") == 9) & (pl.col("rw") == 2))
print(c9.group_by(pl.col("CT_dead") >= 5).agg(pl.len().alias("n"),
                                              pl.col("chain_dis").mean().round(3).alias("chain_dis_mean"),
                                              pl.col("T_dead").mean().round(2).alias("mean_T_dead")))
print("CT_dead distribution among rw=T, label 9 ctrl A:")
print(c9.group_by("CT_dead").agg(pl.len().alias("n")).sort("CT_dead"))

print("\n== C4-3: co-occurrence of chain-based inversions at labels 9 and 13, ctrl_late r1=true, per match ==")
chain_flags = (X.filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & pl.col("chain_ok") &
                        pl.col("round").is_in([8, 9, 10, 11, 13, 14, 15, 16]))
               .select(["match_key", "round", "chain_dis"]))
flags = chain_flags.pivot(on="round", index="match_key", values="chain_dis", aggregate_function="first")
flags = flags.rename({str(c): f"d{c}" for c in [8, 9, 10, 11, 13, 14, 15, 16] if str(c) in flags.columns})
print("matches with chain flags at >=1 label:", flags.height)
for a, bb in [("d9", "d13"), ("d8", "d13"), ("d10", "d13"), ("d9", "d14"), ("d9", "d10")]:
    if a in flags.columns and bb in flags.columns:
        f = flags.drop_nulls([a, bb])
        n11 = f.filter((pl.col(a) == 1) & (pl.col(bb) == 1)).height
        n1x = f.filter(pl.col(a) == 1).height
        nx1 = f.filter(pl.col(bb) == 1).height
        print(f"{a} x {bb}: n={f.height}  {a}=1: {n1x}  {bb}=1: {nx1}  both=1: {n11}  "
              f"P({bb}=1|{a}=1)={round(n11 / max(n1x, 1), 3)}  P({bb}=1|{a}=0)="
              f"{round(f.filter((pl.col(a) == 0) & (pl.col(bb) == 1)).height / max(f.filter(pl.col(a) == 0).height, 1), 3)}")
