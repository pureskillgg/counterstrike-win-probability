"""verify D5-6 check 6: does a header field carry the format behind the flagged halftime round, and does the
all-five-dead test survive excluding the hostage map cs_office?

Header fields used: match_type, map_name, final score sum. Regime A matches only. Pattern labels come from
chain-vs-round_end disagreement (see verify_D5_6_check5.py). Also reports the no-plant T-win contradiction for
label 13 with cs_office excluded.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_6_check6.py
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


H = (pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "match_type", "map_name", "t_starters_score_final", "ct_starters_score_final"]).collect()
     .rename({"key": "match_key"}).with_columns(final_sum=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")))
print("match_type values (all 2024 matches):")
print(H.group_by("match_type").agg(pl.len().alias("n")).sort("n", descending=True))

T = pl.read_parquet(SCR / "v_rounds.parquet")
M = pl.read_parquet(SCR / "v_matches.parquet")
LATE = [f"{d:02d}" for d in range(12, 26)]
M = M.with_columns(grp=pl.when(pl.col("aff_pc20")).then(pl.lit("aff_r13"))
                   .when(pl.col("day").is_in(LATE)).then(pl.lit("ctrl_late")).otherwise(pl.lit("other")))
TT = T.filter(pl.col("re_present") & pl.col("rw").is_not_null()).select(K + ["rw", "chain", "bs_planted", "re_present"])
TT = TT.with_columns(chain_ok=pl.col("chain").is_not_null() & (pl.col("chain") != 0),
                     flag=pl.when(pl.col("chain").is_not_null() & (pl.col("chain") != 0))
                     .then((pl.col("chain") != pl.col("rw")).cast(pl.Int8)).otherwise(None))
A = TT.join(M.filter(pl.col("r1_end")).select(["match_key", "map_name"]), on="match_key", how="inner")
pat = (A.filter(pl.col("flag") == 1).group_by("match_key").agg(pl.col("round").sort().alias("fl")))
pat = pat.with_columns(pattern=pl.col("fl").list.eval(pl.element().cast(pl.Utf8)).list.join(","))
allm = M.filter(pl.col("r1_end")).select(["match_key", "map_name", "aff_pc20"]).join(
    H.select(["match_key", "final_sum"]), on="match_key", how="left").join(
    pat.select(["match_key", "pattern"]), on="match_key", how="left").with_columns(pattern=pl.col("pattern").fill_null("none"))
allm = allm.join(H.select(["match_key", "match_type"]), on="match_key", how="left")
top = ["13", "9", "13,25,28", "13,25,27,28", "13,25", "none"]
print("\n== F1: regime A matches: flag pattern x match_type (top patterns) ==")
print(allm.filter(pl.col("pattern").is_in(top)).group_by(["pattern", "match_type"]).agg(pl.len().alias("n")).sort(
    ["pattern", "n"], descending=[False, True]))

print("\n== F2: regime A matches: flag pattern x final score sum bucket ==")
print(allm.filter(pl.col("pattern").is_in(top)).with_columns(
    bucket=pl.when(pl.col("final_sum") <= 17).then(pl.lit("<=17")).when(pl.col("final_sum") <= 25).then(pl.lit("18-25"))
    .otherwise(pl.lit(">=26"))).group_by(["pattern", "bucket"]).agg(pl.len().alias("n")).sort(["pattern", "bucket"]))

# all-five-dead test excluding cs_office, label 13 and other labels (ctrl late, regime A, no plant)
DE = scan("player_death", ["match_key", "round", "tick", "player_team_code", "player_id"]).filter(
    pl.col("player_team_code").is_in([2, 3]) & pl.col("player_id").is_not_null()).collect()
NP = (T.filter(pl.col("re_present") & pl.col("rw").is_not_null() & (pl.col("bs_planted") == 0))
      .join(M.select(["match_key", "grp", "r1_end", "map_name"]), on="match_key", how="left")
      .filter((pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("map_name") != "cs_office"))
      .select(K + ["rw", "fe_tick", "re_tick", "chain"]))
D = DE.join(NP.select(K + ["fe_tick", "re_tick"]), on=K, how="inner").filter(
    (pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("re_tick")))
ps = (D.group_by(K + ["player_team_code"]).agg(pl.col("player_id").n_unique().alias("n"))
      .pivot(on="player_team_code", index=K, values="n", aggregate_function="first")
      .rename({"2": "T_dead", "3": "CT_dead"}))
X = NP.join(ps, on=K, how="left").with_columns(pl.col("T_dead").fill_null(0), pl.col("CT_dead").fill_null(0))
X = X.filter((pl.col("T_dead") + pl.col("CT_dead")) > 0)
print("\n== F3: no-plant rounds, ctrl late, regime A, cs_office excluded: rw=T with CT all dead, by label ==")
print(X.filter(pl.col("rw") == 2).group_by("round").agg(
    pl.len().alias("n_rwT"), (pl.col("CT_dead") >= 5).sum().alias("CT_all5"),
    (pl.col("CT_dead") >= 5).mean().round(3).alias("share")).sort("round").filter(
    pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 15, 16, 25, 28])))
print("\nrw=CT with T all dead (consistent) vs CT all dead (contradiction), label 13 vs 12/14:")
print(X.filter(pl.col("rw") == 3).group_by("round").agg(
    pl.len().alias("n_rwCT"), (pl.col("CT_dead") >= 5).sum().alias("CT_all5_contradiction"),
    (pl.col("T_dead") >= 5).mean().round(3).alias("share_T_all5")).sort("round").filter(
    pl.col("round").is_in([8, 9, 10, 11, 12, 13, 14, 15, 16])))
print("\ncs_office rows in the label-13 no-plant set (ctrl late, regime A), before exclusion:")
cs = (T.filter(pl.col("re_present") & (pl.col("bs_planted") == 0) & (pl.col("round") == 13)).join(
    M.select(["match_key", "grp", "r1_end", "map_name"]), on="match_key", how="left").filter(
    (pl.col("grp") == "ctrl_late") & pl.col("r1_end") & (pl.col("map_name") == "cs_office")))
print("cs_office label-13 no-plant rows:", cs.height)
